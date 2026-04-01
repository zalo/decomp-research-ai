"""Run Ghidra headless decompiler as a fallback when m2c fails."""

import subprocess
import os
from pathlib import Path
from typing import Optional

from .config import Config


def decompile_with_ghidra(config: Config, func_name: str,
                          source_path: str) -> Optional[str]:
    """Decompile a function using Ghidra headless mode via ghidrecomp.

    Returns the decompiled C code, or None on failure.
    Requires: ghidrecomp, Ghidra 12+, xvfb-run
    """
    # Find the target .o file
    import json
    with open(config.objdiff_json_path) as f:
        data = json.load(f)

    obj_path = None
    for unit in data["units"]:
        if unit.get("metadata", {}).get("source_path") == source_path:
            obj_path = config.melee_root / unit["target_path"]
            break

    if not obj_path or not obj_path.exists():
        return None

    ghidra_dir = Path.home() / "ghidra12"
    if not ghidra_dir.exists():
        return None

    output_dir = Path("/tmp/ghidra_decomp")
    output_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["GHIDRA_INSTALL_DIR"] = str(ghidra_dir)

    try:
        result = subprocess.run(
            ["xvfb-run", "-a", "ghidrecomp", str(obj_path),
             "--output-path", str(output_dir),
             "--filter", func_name],
            capture_output=True, text=True, timeout=120,
            env=env,
        )

        if result.returncode != 0:
            return None

        # Find the output file
        for decomp_file in output_dir.rglob(f"{func_name}*.c"):
            code = decomp_file.read_text().strip()
            if code:
                return code

        return None
    except (subprocess.TimeoutExpired, Exception):
        return None
