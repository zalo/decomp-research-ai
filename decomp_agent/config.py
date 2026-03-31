"""Configuration for the decomp automation agent."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    # Project paths
    melee_root: Path = Path("/home/selstad/Desktop/DecompAgent/melee")
    permuter_root: Path = Path("/home/selstad/Desktop/DecompAgent/decomp-permuter")
    agent_root: Path = Path("/home/selstad/Desktop/DecompAgent/decomp_agent")

    # Derived paths (set in __post_init__)
    report_path: Path = field(init=False)
    objdiff_json_path: Path = field(init=False)
    configure_py_path: Path = field(init=False)
    symbols_path: Path = field(init=False)
    asm_root: Path = field(init=False)
    obj_root: Path = field(init=False)
    src_root: Path = field(init=False)
    ctx_root: Path = field(init=False)
    state_db_path: Path = field(init=False)
    nonmatchings_root: Path = field(init=False)

    # AI configuration (Anthropic direct)
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    api_base_url: str = "https://api.anthropic.com"
    initial_model: str = "claude-opus-4-20250514"
    cheap_model: str = "claude-sonnet-4-20250514"
    expensive_model: str = "claude-opus-4-20250514"

    # Parallelism
    max_workers: int = 4
    permuter_threads: int = 4

    # Limits per function
    max_logic_fix_attempts: int = 10
    max_regalloc_fix_attempts: int = 10
    max_syntax_fix_attempts: int = 10
    permuter_timeout_secs: int = 120
    token_budget_per_func: int = 0  # 0 = unlimited

    # Filtering
    max_function_size: int = 2048
    min_match_for_regalloc: float = 80.0

    def __post_init__(self):
        br = self.melee_root / "build" / "GALE01"
        self.report_path = br / "report.json"
        self.objdiff_json_path = self.melee_root / "objdiff.json"
        self.configure_py_path = self.melee_root / "configure.py"
        self.symbols_path = self.melee_root / "config" / "GALE01" / "symbols.txt"
        self.asm_root = br / "asm"
        self.obj_root = br / "obj"
        self.src_root = self.melee_root / "src"
        self.ctx_root = br / "src"
        self.state_db_path = self.melee_root / "decomp_agent_state.db"
        self.nonmatchings_root = self.melee_root / "nonmatchings"
