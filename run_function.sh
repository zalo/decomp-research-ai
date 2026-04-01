#!/bin/bash
# Run the decomp agent on a single function and verify the result
# Usage: ./run_function.sh <function_name> [max_attempts]
# Example: ./run_function.sh grIceMt_801F929C 10

FUNC=${1:?Usage: $0 <function_name> [max_attempts]}
MAX=${2:-10}

export ANTHROPIC_API_KEY=""$ANTHROPIC_API_KEY""
cd /home/selstad/Desktop/DecompAgent

rm -f melee/decomp_agent_state.db
timeout 300 python3 -u -c "
from decomp_agent.config import Config
from decomp_agent.state_db import StateDB
from decomp_agent.target_selector import populate_db
from decomp_agent.orchestrator import process_function
config = Config()
config.max_logic_fix_attempts = $MAX
db = StateDB(config.state_db_path)
populate_db(config, db)
fs = db.get_state('$FUNC')
if not fs:
    print('Function not found in DB')
    exit(1)
print(f'=== {fs.func_name} at {fs.initial_match_pct}% ===')
result = process_function(config, db, fs)
after = db.get_state(fs.func_name)
pct = f'{after.current_match_pct:.1f}%' if after.current_match_pct else 'n/a'
print(f'\nResult: {result}, match: {pct}')
db.close()
"

# Verify build
echo ""
echo "=== Build check ==="
cd melee
BUILD=$(ninja 2>&1)
if echo "$BUILD" | tail -1 | grep -q "Trophies"; then
    echo "BUILD: PASS"
    echo "$BUILD" | grep "matched"
    echo ""
    echo "Changed files:"
    git diff --name-only
else
    echo "BUILD: FAIL"
    echo "$BUILD" | grep "Error:" | tail -3
    echo ""
    echo "Reverting..."
    git checkout -- .
    find src/ -name '*.agent_bak' -delete 2>/dev/null
    ninja > /dev/null 2>&1
    echo "Restored to last commit"
fi
