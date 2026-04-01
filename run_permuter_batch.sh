#!/bin/bash
# Run the permuter on functions at 95%+ fuzzy match (regalloc-only)
# These are the best candidates for automated register allocation fixing
# Usage: ./run_permuter_batch.sh [count] [timeout_per_func]
COUNT=${1:-10}
TIMEOUT=${2:-300}  # 5 min per function

export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
cd /home/selstad/Desktop/DecompAgent
LOG=/tmp/decomp_permuter_$(date +%H%M%S).log

echo "=== Permuter batch: $COUNT functions, ${TIMEOUT}s timeout ===" | tee $LOG

python3 -u << PYEOF 2>&1 | tee -a $LOG
import json, subprocess
from pathlib import Path
from decomp_agent.config import Config
from decomp_agent.permuter_runner import permute_function
from decomp_agent.build_diff import build_and_diff, diff_function

config = Config()
config.permuter_timeout_secs = $TIMEOUT
MELEE = Path("melee")

# Find 95%+ fuzzy functions that are regalloc-only
with open(MELEE / "build/GALE01/report.json") as f:
    report = json.load(f)
with open(MELEE / "objdiff.json") as f:
    units = {u["name"]: u for u in json.load(f)["units"]}

candidates = []
for unit in report.get("units", []):
    meta = units.get(unit["name"], {}).get("metadata", {})
    if meta.get("complete"): continue
    src = meta.get("source_path", "")
    if not src: continue
    for func in unit.get("functions", []):
        pct = func.get("fuzzy_match_percent")
        size = int(func.get("size", "0"))
        if pct and 95 <= pct < 100 and 20 < size <= 500:
            candidates.append((pct, size, func["name"], unit["name"], src))

candidates.sort(key=lambda x: (-x[0], x[1]))
print(f"Found {len(candidates)} candidates at 95%+")

matched = 0
for i, (pct, size, name, unit_name, src) in enumerate(candidates[:$COUNT]):
    print(f"\n[{i+1}/$COUNT] {name} ({pct:.1f}%, {size}b)")

    result = permute_function(config, name, unit_name, src)

    if result.matched:
        print(f"  PERMUTER MATCHED! score=0")
        # Verify build
        build = subprocess.run(["ninja"], capture_output=True, text=True, cwd=str(MELEE))
        if "Trophies" in build.stdout:
            matched += 1
            subprocess.run(["git", "add", "-A"], cwd=str(MELEE))
            subprocess.run(["git", "commit", "-m",
                f"Match {name} via permuter\n\nCo-Authored-By: decomp-permuter"],
                cwd=str(MELEE), capture_output=True)
            print(f"  COMMITTED!")
        else:
            print(f"  Build failed, reverting")
            subprocess.run(["git", "checkout", "--", "."], cwd=str(MELEE), capture_output=True)
    else:
        best = result.best_score
        print(f"  Best score: {best} (not matched)")

print(f"\n=== DONE: {matched} matched via permuter ===")
PYEOF

echo "Finished at $(date)" | tee -a $LOG
