"""Main orchestrator for the automated decompilation pipeline.

Pipeline per function:
  1. DECOMPILE: If placeholder exists, run m2c. If m2c fails, try AI.
  2. BUILD: Compile the .o file via ninja.
  3. DIFF: Compare against target via objdiff-cli.
  4. CLASSIFY: Determine what kind of fix is needed.
  5. FIX: Run permuter (regalloc) or AI (logic/syntax).
  6. REPORT: Log before/after match percentages.

On ANY unexpected failure, the pipeline stops early with a clear error
message so we can debug and add a handler, rather than silently skipping.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Config
from .state_db import StateDB, FunctionState, Attempt, TERMINAL_STATES
from .target_selector import populate_db, load_report
from .build_diff import build_and_diff, DiffResult
from .source_editor import (extract_function_asm, extract_function_c,
                            has_placeholder, get_nearby_matched_c)
from .safe_edit import SafeFile
from .m2c_runner import decompile_function
from .permuter_runner import permute_function
from .ai_fixer import (initial_decompile, fix_logic, fix_regalloc, fix_syntax,
                       ConversationAgent)
from .prompts import MWCC_CONTEXT, LOGIC_FIX


class PipelineError(Exception):
    """Raised when an unexpected failure occurs that needs human attention."""
    pass


def classify(diff: DiffResult) -> str:
    if not diff.compiled_ok:
        return "COMPILE_ERROR"
    if diff.is_matched:
        return "MATCHED"
    if diff.is_regalloc_only:
        return "REGALLOC_ONLY"
    return "SIZE_MISMATCH"


# ---------------------------------------------------------------------------
# Step 1: Ensure the function has C code (m2c or AI decompile)
# ---------------------------------------------------------------------------

def _ensure_c_code(config, db, func, unit, src, asm, dry_run):
    """Make sure the function has C code in the source file.

    Returns True if code exists (or was just created), False to skip.
    Raises PipelineError on unexpected failures.
    """
    src_path = config.melee_root / src

    if not has_placeholder(src_path, func):
        # Function already has code (or no placeholder to fill)
        if extract_function_c(src_path, func) is not None:
            return True  # already has code
        # No placeholder AND no code — can't do anything
        print(f"    → No placeholder and no existing code, skipping")
        db.update_state(func, "SKIPPED", last_error="no_placeholder_no_code")
        return False

    if dry_run:
        print(f"    → Would decompile {func}")
        return False

    db.update_state(func, "DECOMPILING")

    # Try m2c (free, fast)
    last_error = ""
    c_code = decompile_function(config, func, src, asm)
    if c_code:
        sf = SafeFile(config, src, unit)
        edit = sf.write_placeholder(func, c_code)
        if edit.success:
            print(f"    → m2c: {len(c_code)} chars, compiles OK")
            return True
        else:
            last_error = edit.compile_error or ""
            print(f"    → m2c doesn't compile: {last_error[:150]}")

    # m2c failed or didn't compile — try AI with error feedback
    target_asm = _get_target_asm(config, func, src, asm)
    if not target_asm:
        print(f"    → No assembly found for {func}")
        db.update_state(func, "SKIPPED", last_error="no_asm_found")
        return False

    nearby_c, includes = _get_context(config, func, unit, src)
    max_decompile_attempts = min(3, config.max_syntax_fix_attempts)
    print(f"    → Trying AI decompile (up to {max_decompile_attempts} attempts)...")

    for attempt in range(max_decompile_attempts):
        # Build prompt with error feedback from previous attempt
        if last_error and attempt > 0:
            error_hint = f"\n\nPREVIOUS ATTEMPT FAILED TO COMPILE:\n{last_error[:500]}\n\nFix the error and try again."
        elif last_error and c_code:
            error_hint = f"\n\nm2c GENERATED THIS (doesn't compile):\n```c\n{c_code[:500]}\n```\nERROR: {last_error[:300]}\n\nFix it or rewrite from scratch."
        else:
            error_hint = ""

        ai_result = initial_decompile(config, target_asm + error_hint,
                                      nearby_c=nearby_c, includes=includes)
        if not ai_result.success or not ai_result.c_code:
            print(f"    → attempt {attempt+1}/{config.max_syntax_fix_attempts}: no valid code")
            continue
        # Reject trivially short responses (just "return;" etc)
        if len(ai_result.c_code.strip()) < 50:
            print(f"    → attempt {attempt+1}/{config.max_syntax_fix_attempts}: response too short ({len(ai_result.c_code)} chars), skipping")
            continue

        db.log_attempt(Attempt(
            func_name=func, state="DECOMPILING", match_pct=None,
            diff_summary=f"ai_decompile_attempt_{attempt+1}",
            ai_model=ai_result.model,
            ai_prompt_tokens=ai_result.prompt_tokens,
            ai_response_tokens=ai_result.response_tokens, duration_secs=0,
        ))

        sf = SafeFile(config, src, unit)
        edit = sf.write_placeholder(func, ai_result.c_code)
        if edit.success:
            print(f"    → attempt {attempt+1}/{config.max_syntax_fix_attempts}: compiles! ({len(ai_result.c_code)} chars)")
            return True
        else:
            last_error = edit.compile_error or ""
            print(f"    → attempt {attempt+1}/{config.max_syntax_fix_attempts}: compile error: {last_error[:150]}")

    print(f"    → All {max_decompile_attempts} decompile attempts failed")
    db.update_state(func, "SKIPPED", last_error=f"decompile_failed: {last_error[:100]}")
    return False


# ---------------------------------------------------------------------------
# Step 2-4: Build, diff, classify
# ---------------------------------------------------------------------------

def _build_and_classify(config, db, func, unit, src):
    """Build the unit and classify the diff. Returns (DiffResult, category)."""
    diff = build_and_diff(config, src, unit, func)

    if not diff.compiled_ok:
        print(f"    → COMPILE_ERROR: {diff.compile_error[:200] if diff.compile_error else 'unknown'}")

    category = classify(diff)
    match_pct = diff.func_match_pct

    db.update_state(func, category, current_match_pct=match_pct)
    db.increment(func, "attempt_count")
    db.log_attempt(Attempt(
        func_name=func, state=category, match_pct=match_pct,
        diff_summary=diff.summary, ai_model=None,
        ai_prompt_tokens=None, ai_response_tokens=None,
        duration_secs=diff.duration_secs,
    ))

    print(f"    → {diff.summary}")
    return diff, category


# ---------------------------------------------------------------------------
# Step 5: Fix based on classification
# ---------------------------------------------------------------------------

def _handle_regalloc(config, db, func, unit, src, diff):
    """Permuter first, then AI for register allocation."""
    # Try permuter
    db.update_state(func, "PERMUTER")
    print(f"    → Running permuter ({config.permuter_timeout_secs}s)...")
    perm_result = permute_function(config, func, unit, src)
    db.update_state(func, "PERMUTER", permuter_best_score=perm_result.best_score)

    if perm_result.matched and perm_result.best_source:
        sf = SafeFile(config, src, unit)
        edit = sf.write_function(func, perm_result.best_source,
                                 min_match_pct=diff.func_match_pct or 0)
        if edit.success:
            new_diff = build_and_diff(config, src, unit, func)
            if new_diff.is_matched:
                db.update_state(func, "MATCHED", current_match_pct=100.0)
                print(f"    → MATCHED via permuter!")
                return "MATCHED"

    # Permuter didn't solve it — try AI regalloc fixes
    fs = db.get_state(func)
    for attempt in range(config.max_regalloc_fix_attempts):
        if fs.regalloc_fix_attempts >= config.max_regalloc_fix_attempts:
            break

        current_c = extract_function_c(config.melee_root / src, func) or ""
        result = fix_regalloc(config, current_c, diff.summary)

        if not result.success or not result.c_code:
            print(f"    → AI regalloc attempt {attempt+1}: no valid code")
            break

        sf = SafeFile(config, src, unit)
        edit = sf.write_function(func, result.c_code,
                                 min_match_pct=diff.func_match_pct or 0)
        if edit.success:
            db.increment(func, "regalloc_fix_attempts")
            db.log_attempt(Attempt(
                func_name=func, state="AI_REGALLOC_FIX", match_pct=None,
                diff_summary="regalloc_fix", ai_model=result.model,
                ai_prompt_tokens=result.prompt_tokens,
                ai_response_tokens=result.response_tokens, duration_secs=0,
            ))
            new_diff = build_and_diff(config, src, unit, func)
            if new_diff.is_matched:
                db.update_state(func, "MATCHED", current_match_pct=100.0)
                print(f"    → MATCHED via AI regalloc!")
                return "MATCHED"
            print(f"    → AI regalloc attempt {attempt+1}: {new_diff.summary}")
            diff = new_diff
        else:
            print(f"    → AI regalloc attempt {attempt+1}: {edit.compile_error[:100] if edit.compile_error else 'failed'}")

        fs = db.get_state(func)

    # Keep whatever progress we have
    pct = diff.func_match_pct
    db.update_state(func, "IMPROVED", current_match_pct=pct)
    print(f"    → Kept at {pct}%")
    return "IMPROVED"


def _handle_size_mismatch(config, db, func, unit, src, asm, diff):
    """AI agent with tools fixes logic/size mismatches."""
    from .tools import ToolExecutor

    executor = ToolExecutor(config, func, unit, src, asm)
    executor.best_match_pct = diff.func_match_pct or 0  # don't accept worse than current
    agent = ConversationAgent(config, MWCC_CONTEXT, tool_executor=executor)

    # Pre-load ALL context so the agent can start working immediately
    target_asm = executor._get_target_assembly()
    current_c = executor._get_current_c_code()
    nearby = executor._get_nearby_functions()
    includes = executor._get_includes()
    m2c_output = executor._run_m2c()

    task = (
        f"Match the function `{func}` for GameCube Super Smash Bros. Melee.\n\n"
        f"STATUS: {diff.summary}\n"
        f"Target: {diff.func_size_target} bytes | Current: {diff.func_size_compiled} bytes\n\n"
        f"=== TARGET ASSEMBLY ===\n{target_asm}\n\n"
        f"=== M2C DECOMPILER OUTPUT (structurally correct, needs type fixes) ===\n{m2c_output}\n\n"
        f"=== CURRENT C CODE IN REPO (may be wrong) ===\n{current_c}\n\n"
        f"=== NEARBY MATCHED FUNCTIONS (same file, correct style) ===\n{nearby}\n\n"
        f"=== INCLUDES ===\n{includes}\n\n"
        f"CRITICAL: Keep the EXACT same function signature (return type, name, params) as:\n"
        f"  {current_c.split(chr(10))[0] if current_c else '(see current code)'}\n"
        f"Changing the signature WILL break other functions that call this one.\n\n"
        f"WORKFLOW:\n"
        f"1. m2c output has correct LOGIC but wrong types. Start from it.\n"
        f"2. Use search_struct_field to map unkXX offsets to real field names\n"
        f"3. Use compile_and_diff to test your code\n"
        f"4. Use side_by_side_diff to see EXACTLY which instructions differ\n"
        f"5. If target is BIGGER: add dead reads (ip->field;) or PAD_STACK\n"
        f"6. If your code is BIGGER: remove temps, casts, use direct access\n"
        f"7. The current repo code may be wrong — trust m2c over it\n\n"
        f"When done, respond with your best C code in a ```c block."
    )

    print(f"    -> Starting agentic loop (bail after 5 stale compiles)...")
    result = agent.run(task, max_tool_rounds=config.max_logic_fix_attempts * 3)

    # Clean up backup file
    bak = (config.melee_root / src).with_suffix(".c.agent_bak")
    if bak.exists():
        bak.unlink()

    # Apply the best code the tool found ONLY if it strictly improved
    original_pct = diff.func_match_pct or 0
    if (executor.best_code and executor.best_match_pct > original_pct
            and executor.best_match_pct > 0):
        from .source_editor import replace_function
        replace_function(config.melee_root / src, func, executor.best_code)
        from .build_diff import build_unit as _bu
        _bu(config, src, unit)
        print(f"    -> Applied best code from tool ({executor.best_match_pct:.1f}%)")

    # Check current state
    final_diff = build_and_diff(config, src, unit, func)
    final_cat = classify(final_diff)
    final_pct = final_diff.func_match_pct

    db.log_attempt(Attempt(
        func_name=func, state="AI_LOGIC_FIX", match_pct=final_pct,
        diff_summary=final_diff.summary, ai_model=config.initial_model,
        ai_prompt_tokens=agent.total_prompt_tokens,
        ai_response_tokens=agent.total_response_tokens,
        duration_secs=0,
    ))

    # If the agent's tool calls achieved a match, great
    if final_cat == "MATCHED":
        db.update_state(func, "MATCHED", current_match_pct=100.0)
        print(f"    -> MATCHED via agentic AI! ({diff.func_match_pct}% -> 100%)")
        return "MATCHED"

    # The tool loop already tracked and applied the best code above.
    # Don't try the agent's final text response — it may be worse.

    if final_cat == "REGALLOC_ONLY":
        print(f"    -> Logic fixed! Now regalloc only. {final_diff.summary}")
        return _handle_regalloc(config, db, func, unit, src, final_diff)

    # Re-diff to get authoritative current state after all tool cleanup
    actual_diff = build_and_diff(config, src, unit, func)
    actual_pct = actual_diff.func_match_pct or 0
    original_pct = diff.func_match_pct or 0

    if actual_pct > original_pct:
        state = "IMPROVED"
    elif actual_pct >= original_pct:
        state = "IMPROVED"  # kept same — still save the agent's work
    else:
        state = "SKIPPED"

    db.update_state(func, state, current_match_pct=actual_pct)
    print(f"    -> {state}: {original_pct:.1f}% -> {actual_pct:.1f}%")
    return state

def _handle_compile_error(config, db, func, unit, src, diff):
    """AI syntax fixes for compilation errors."""
    for attempt in range(config.max_syntax_fix_attempts):
        current_c = extract_function_c(config.melee_root / src, func) or ""
        result = fix_syntax(config, current_c, diff.compile_error or "")

        if not result.success or not result.c_code:
            print(f"    → AI syntax attempt {attempt+1}: no valid code")
            break

        sf = SafeFile(config, src, unit)
        edit = sf.write_function(func, result.c_code)
        if edit.success:
            db.log_attempt(Attempt(
                func_name=func, state="AI_SYNTAX_FIX", match_pct=None,
                diff_summary="syntax_fix", ai_model=result.model,
                ai_prompt_tokens=result.prompt_tokens,
                ai_response_tokens=result.response_tokens, duration_secs=0,
            ))
            new_diff = build_and_diff(config, src, unit, func)
            if new_diff.compiled_ok:
                print(f"    → AI syntax attempt {attempt+1}: now compiles! {new_diff.summary}")
                return classify(new_diff), new_diff
            print(f"    → AI syntax attempt {attempt+1}: still doesn't compile")
            diff = new_diff
        else:
            print(f"    → AI syntax attempt {attempt+1}: {edit.compile_error[:100] if edit.compile_error else 'failed'}")

    db.update_state(func, "SKIPPED", last_error="compile_error_unresolved")
    return "SKIPPED", diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_target_asm(config, func, src, asm):
    """Get the target assembly for a function, trying multiple paths."""
    target_asm = extract_function_asm(Path(config.melee_root / asm), func)
    if not target_asm:
        asm_alt = config.asm_root / src.removeprefix("src/").replace(".c", ".s")
        target_asm = extract_function_asm(asm_alt, func)
    return target_asm or ""


def _get_context(config, func, unit, src):
    """Get nearby matched C code and includes for AI context."""
    src_path = config.melee_root / src
    nearby_c = ""
    report = load_report(config)
    for u in report.get("units", []):
        if u["name"] == unit:
            nearby = get_nearby_matched_c(src_path, u.get("functions", []), func)
            if nearby:
                nearby_c = nearby[0][1]
            break

    includes = ""
    if src_path.exists():
        lines = src_path.read_text().split("\n")
        includes = "\n".join(l for l in lines[:30] if l.startswith("#include"))

    return nearby_c, includes


# ---------------------------------------------------------------------------
# Main process_function: clean state machine
# ---------------------------------------------------------------------------

def process_function(config: Config, db: StateDB, fs: FunctionState,
                     dry_run: bool = False) -> str:
    """Process a single function. Returns final state string."""
    func = fs.func_name
    unit = fs.unit_name
    src = fs.source_path
    asm = fs.asm_path

    pct_str = f"{fs.initial_match_pct:.1f}%" if fs.initial_match_pct is not None else "new"
    print(f"  [{func}] match={pct_str} size={fs.size_bytes}b")

    # Step 1: Ensure C code exists
    if not _ensure_c_code(config, db, func, unit, src, asm, dry_run):
        return db.get_state(func).state if db.get_state(func) else "SKIPPED"

    if dry_run:
        return "BUILDING"

    # Step 2-4: Build, diff, classify
    diff, category = _build_and_classify(config, db, func, unit, src)

    if category == "MATCHED":
        return "MATCHED"

    # Step 5: Fix based on classification
    if category == "COMPILE_ERROR":
        category, diff = _handle_compile_error(config, db, func, unit, src, diff)
        if category in ("SKIPPED", "MATCHED"):
            return category
        # Re-classify after syntax fix
        category = classify(diff)

    if category == "REGALLOC_ONLY":
        return _handle_regalloc(config, db, func, unit, src, diff)

    if category == "SIZE_MISMATCH":
        return _handle_size_mismatch(config, db, func, unit, src, asm, diff)

    return category


# ---------------------------------------------------------------------------
# Orchestrator main loop
# ---------------------------------------------------------------------------

def run(config: Config, limit: int = 0, dry_run: bool = False,
        workers: int = 1):
    db = StateDB(config.state_db_path)

    print("Populating database from report.json...")
    count = populate_db(config, db)
    print(f"  {count} functions loaded\n")

    pending = db.get_pending(batch_size=limit if limit else 9999)
    if not pending:
        print("No pending functions to process.")
        db.close()
        return

    # Group by unit, one function per unit for max parallelism
    by_unit = {}
    seen = set()
    for fs in pending:
        if fs.func_name not in seen:
            seen.add(fs.func_name)
            by_unit.setdefault(fs.unit_name, []).append(fs)

    all_results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        processed = 0
        for unit_name, funcs in by_unit.items():
            remaining = (limit - processed) if limit else len(funcs)
            if remaining <= 0:
                break
            batch = funcs[:remaining]
            processed += len(batch)
            futures[executor.submit(
                _process_unit_batch, config, db, unit_name, batch, dry_run
            )] = unit_name

        for future in as_completed(futures):
            unit_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"\n  FATAL ERROR in unit {unit_name}: {e}")
                print(f"  Stopping early to debug.")
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    # Print summary
    _print_summary(db, all_results)
    db.close()


def _process_unit_batch(config, db, unit_name, funcs, dry_run):
    results = []
    print(f"\n[Unit: {unit_name}]")
    for fs in funcs:
        before_pct = fs.initial_match_pct
        try:
            final_state = process_function(config, db, fs, dry_run)
            after_fs = db.get_state(fs.func_name)
            after_pct = after_fs.current_match_pct if after_fs else None
            results.append((fs.func_name, final_state, before_pct, after_pct))
        except PipelineError as e:
            print(f"\n  PIPELINE ERROR {fs.func_name}: {e}")
            print(f"  This needs a fix. Stopping this unit.")
            db.update_state(fs.func_name, "FAILED", last_error=str(e)[:200])
            results.append((fs.func_name, "FAILED", before_pct, None))
            raise  # propagate to stop the whole run
        except Exception as e:
            print(f"  ERROR {fs.func_name}: {e}")
            db.update_state(fs.func_name, "FAILED", last_error=str(e)[:200])
            results.append((fs.func_name, "FAILED", before_pct, None))
    return results


def _print_summary(db, all_results):
    stats = db.get_stats()
    print(f"\n{'='*72}")
    print(f"FINAL SUMMARY")
    print(f"{'='*72}")
    for state, count in sorted(stats.items()):
        if count > 0:
            print(f"  {state}: {count}")

    if all_results:
        print(f"\n  {'Function':<40} {'Before':>7} {'After':>7} {'Delta':>7}  Result")
        print(f"  {'-'*75}")
        for func_name, state, before, after in all_results:
            bs = f"{before:.1f}%" if before is not None else "new"
            astr = f"{after:.1f}%" if after is not None else "n/a"
            d = (after or 0) - (before or 0)
            ds = f"{'+' if d >= 0 else ''}{d:.1f}%"
            icon = "✓" if state == "MATCHED" else "✗" if state in ("SKIPPED","FAILED") else "↑"
            print(f"  {icon} {func_name:<38} {bs:>7} {astr:>7} {ds:>7}  {state}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Automated decomp agent")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = Config()
    run(config, limit=args.limit, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
