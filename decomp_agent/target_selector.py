"""Parse report.json and rank functions for decompilation."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .config import Config
from .state_db import StateDB


@dataclass
class TargetFunction:
    func_name: str
    unit_name: str
    source_path: str
    asm_path: str
    size_bytes: int
    match_pct: Optional[float]
    score: float  # higher = better candidate
    is_placeholder: bool = False  # no C code yet, just /// #funcname


def load_report(config: Config) -> dict:
    with open(config.report_path) as f:
        return json.load(f)


def load_objdiff_units(config: Config) -> dict:
    """Load objdiff.json and index by unit name."""
    with open(config.objdiff_json_path) as f:
        data = json.load(f)
    return {u["name"]: u for u in data["units"]}


def find_unmatched_functions(config: Config) -> list[TargetFunction]:
    """Find all functions that aren't 100% matched, ranked by expected difficulty."""
    report = load_report(config)
    objdiff_units = load_objdiff_units(config)
    targets = []

    for unit in report.get("units", []):
        unit_name = unit["name"]
        unit_info = objdiff_units.get(unit_name, {})
        metadata = unit_info.get("metadata", {})

        # Skip completed units
        if metadata.get("complete", False):
            continue

        source_path = metadata.get("source_path", "")
        if not source_path:
            continue

        # Derive asm path from unit name
        # unit_name: "main/melee/lb/lbarq" -> asm: "build/GALE01/asm/melee/lb/lbarq.s"
        unit_rel = unit_name.removeprefix("main/")
        asm_path = str(config.asm_root / f"{unit_rel}.s")

        # Count how many functions in this unit are matched (for context bonus)
        funcs = unit.get("functions", [])
        matched_count = sum(1 for f in funcs if f.get("fuzzy_match_percent") == 100.0)
        total_count = len(funcs)
        unit_maturity = matched_count / max(total_count, 1)

        for func in funcs:
            match_pct = func.get("fuzzy_match_percent")
            size = int(func.get("size", "0"))
            name = func.get("name", "")

            if not name or size == 0:
                continue
            if match_pct == 100.0:
                continue
            if size > config.max_function_size:
                continue

            # Check what state the function's source is in
            src_full = config.src_root.parent / source_path
            from .source_editor import has_placeholder, extract_function_c
            has_code = extract_function_c(src_full, name) is not None if src_full.exists() else False
            is_placeholder = has_placeholder(src_full, name) if src_full.exists() else False

            # Skip functions with no code and no placeholder — can't work on them
            if not has_code and not is_placeholder:
                continue

            # Scoring: prefer functions with existing code (can improve immediately)
            size_score = 1.0 / max(size, 1)

            if has_code and match_pct is not None:
                # Has code — prioritize by room for improvement
                if match_pct < 50:
                    match_bonus = 6.0  # lots of room
                elif match_pct < 80:
                    match_bonus = 5.0  # good room
                elif match_pct < 95:
                    match_bonus = 4.0  # moderate room
                else:
                    match_bonus = 2.0  # diminishing returns
            elif is_placeholder:
                match_bonus = 3.0  # needs decompile from scratch
            else:
                match_bonus = 1.0  # has code but no match data

            unit_bonus = 1.0 + unit_maturity  # 1.0-2.0 based on how complete the unit is

            score = size_score * match_bonus * unit_bonus * 10000

            targets.append(TargetFunction(
                func_name=name,
                unit_name=unit_name,
                source_path=source_path,
                asm_path=asm_path,
                size_bytes=size,
                match_pct=match_pct,
                score=score,
                is_placeholder=is_placeholder and not has_code,
            ))

    targets.sort(key=lambda t: t.score, reverse=True)
    return targets


def populate_db(config: Config, db: StateDB):
    """Populate the state database with all unmatched functions."""
    targets = find_unmatched_functions(config)
    count = 0
    for t in targets:
        existing = db.get_state(t.func_name)
        if existing and existing.state in ("MATCHED", "SKIPPED", "FAILED"):
            continue
        db.upsert_function(
            func_name=t.func_name,
            unit_name=t.unit_name,
            source_path=t.source_path,
            asm_path=t.asm_path,
            size_bytes=t.size_bytes,
            initial_match_pct=t.match_pct,
        )
        count += 1
    return count


def main():
    """CLI: show top targets."""
    config = Config()
    targets = find_unmatched_functions(config)

    print(f"Total unmatched functions (≤{config.max_function_size}b): {len(targets)}")
    print()
    print(f"{'Score':>8}  {'Match%':>7}  {'Size':>6}  {'Type':>5}  {'Function':<40}  Unit")
    print("-" * 120)
    for t in targets[:30]:
        pct = f"{t.match_pct:.1f}%" if t.match_pct is not None else "asm"
        typ = "NEW" if t.is_placeholder else "WIP"
        print(f"{t.score:8.1f}  {pct:>7}  {t.size_bytes:>6}  {typ:>5}  {t.func_name:<40}  {t.unit_name}")

    # Also populate DB
    db = StateDB(config.state_db_path)
    count = populate_db(config, db)
    print(f"\nPopulated {count} functions into state DB at {config.state_db_path}")
    stats = db.get_stats()
    print(f"DB stats: {dict(stats)}")
    db.close()


if __name__ == "__main__":
    main()
