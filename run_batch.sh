#!/bin/bash
# Run N functions sequentially, verify build, commit ONLY true improvements
# Usage: ./run_batch.sh [count] [max_attempts]
COUNT=${1:-10}
MAX=${2:-10}

export ANTHROPIC_API_KEY=""$ANTHROPIC_API_KEY""
cd /home/selstad/Desktop/DecompAgent
LOG=/tmp/decomp_batch_$(date +%H%M%S).log

echo "=== Batch: $COUNT functions, $MAX attempts ===" | tee $LOG
echo "Started at $(date)" | tee -a $LOG

# DB persists across batches for state tracking

python3 -u << 'PYEOF' 2>&1 | tee -a $LOG
import subprocess, sys, os
from pathlib import Path
from decomp_agent.config import Config
from decomp_agent.state_db import StateDB
from decomp_agent.target_selector import populate_db
from decomp_agent.orchestrator import process_function
from decomp_agent.build_diff import build_and_diff

COUNT = int(os.environ.get("COUNT", "10"))
MAX = int(os.environ.get("MAX", "10"))

config = Config()
config.max_logic_fix_attempts = MAX
db = StateDB(config.state_db_path)
populate_db(config, db)

MELEE = Path("melee")
matched = 0
improved = 0
skipped = 0

for i in range(COUNT):
    pending = db.get_pending(batch_size=1)
    if not pending:
        print("No more functions")
        break

    fs = pending[0]
    pct_str = f"{fs.initial_match_pct:.1f}%" if fs.initial_match_pct else "new"
    print(f"\n[{i+1}/{COUNT}] {fs.func_name} ({pct_str}, {fs.size_bytes}b)")

    # Measure ACTUAL match BEFORE agent runs (not fuzzy from report)
    pre_diff = build_and_diff(config, fs.source_path, fs.unit_name, fs.func_name)
    pre_pct = pre_diff.func_match_pct or 0

    try:
        result = process_function(config, db, fs)
    except Exception as e:
        print(f"  ERROR: {e}")
        db.update_state(fs.func_name, "FAILED", last_error=str(e)[:200])
        result = "FAILED"

    # Measure ACTUAL match AFTER
    post_diff = build_and_diff(config, fs.source_path, fs.unit_name, fs.func_name)
    post_pct = post_diff.func_match_pct or 0

    # Check if files changed
    changed = subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True, cwd=str(MELEE)).stdout.strip()
    if not changed:
        print(f"  → No changes ({result}) [{pre_pct:.1f}% → {post_pct:.1f}%]")
        skipped += 1
        continue

    # Build check
    build = subprocess.run(["ninja"], capture_output=True, text=True, cwd=str(MELEE))
    build_ok = "Trophies" in build.stdout or "Event Matches" in build.stdout

    if not build_ok:
        print(f"  → BUILD FAIL, reverting")
        subprocess.run(["git", "checkout", "--", "."], cwd=str(MELEE), capture_output=True)
        for bak in MELEE.rglob("*.agent_bak"):
            bak.unlink()
        subprocess.run(["ninja"], cwd=str(MELEE), capture_output=True)
        continue

    # STRICT comparison: post must be BETTER than pre
    if post_pct >= 100.0:
        matched += 1
        subprocess.run(["git", "add", "-A"], cwd=str(MELEE))
        subprocess.run(["git", "commit", "-m",
            f"Match {fs.func_name}\n\nCo-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"],
            cwd=str(MELEE), capture_output=True)
        print(f"  → MATCHED! {pre_pct:.1f}% → 100% COMMITTED")
    elif post_pct > pre_pct + 0.5:  # must improve by at least 0.5%
        improved += 1
        subprocess.run(["git", "add", "-A"], cwd=str(MELEE))
        subprocess.run(["git", "commit", "-m",
            f"Improve {fs.func_name}: {pre_pct:.1f}% -> {post_pct:.1f}%\n\nCo-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"],
            cwd=str(MELEE), capture_output=True)
        print(f"  → IMPROVED {pre_pct:.1f}% → {post_pct:.1f}% COMMITTED")
    else:
        # Same or worse — revert
        subprocess.run(["git", "checkout", "--", "."], cwd=str(MELEE), capture_output=True)
        print(f"  → No real improvement ({pre_pct:.1f}% → {post_pct:.1f}%), reverted")
        skipped += 1

print(f"\n=== DONE: {matched} matched, {improved} improved, {skipped} skipped ===")
progress = subprocess.run(["ninja"], capture_output=True, text=True, cwd=str(MELEE))
for line in progress.stdout.split("\n"):
    if "matched" in line: print(line)
db.close()
PYEOF

echo "Finished at $(date)" | tee -a $LOG
