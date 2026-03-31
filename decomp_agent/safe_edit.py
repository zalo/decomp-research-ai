"""Safe source file editing with backup/restore on build failure."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

from .config import Config
from .build_diff import build_unit


@dataclass
class EditResult:
    success: bool
    compile_error: Optional[str] = None  # error message if build failed


@contextmanager
def safe_source_edit(config: Config, source_path: str, unit_name: str):
    """Context manager that backs up a source file and restores it if the build fails.

    Usage:
        with safe_source_edit(config, "src/melee/lb/lbarq.c", "main/melee/lb/lbarq") as sf:
            sf.write_function(func_name, new_code)
            if not sf.build_ok:
                # file was auto-restored
    """
    yield SafeFile(config, source_path, unit_name)


class SafeFile:
    def __init__(self, config: Config, source_path: str, unit_name: str):
        self.config = config
        self.source_path = source_path
        self.unit_name = unit_name
        self.full_path = config.melee_root / source_path
        self.backup_path = self.full_path.with_suffix(".c.bak")
        self.build_ok = True
        self._backed_up = False

    def _backup(self):
        if not self._backed_up:
            shutil.copy2(self.full_path, self.backup_path)
            self._backed_up = True

    def _restore(self):
        if self._backed_up and self.backup_path.exists():
            shutil.copy2(self.backup_path, self.full_path)
            self.backup_path.unlink()
            self._backed_up = False

    def cleanup(self):
        if self._backed_up and self.backup_path.exists():
            self.backup_path.unlink()
            self._backed_up = False

    def write_function(self, func_name: str, new_code: str,
                       min_match_pct: float = 0.0) -> EditResult:
        """Replace a function in the source file, verify build, restore on failure.

        Returns EditResult with success=True if edit compiled (and didn't degrade).
        Returns compile_error string if build failed, so caller can feed it back to AI.
        """
        from .source_editor import replace_function, extract_function_c

        if not new_code or not new_code.strip():
            return EditResult(False, "empty code")

        self._backup()

        old_code = extract_function_c(self.full_path, func_name)
        if old_code is None:
            self._restore()
            return EditResult(False, "could not find function in source")

        if not replace_function(self.full_path, func_name, new_code):
            self._restore()
            return EditResult(False, "could not replace function in source")

        ok, err = build_unit(self.config, self.source_path, self.unit_name)
        if not ok:
            self._restore()
            self.build_ok = False
            return EditResult(False, err)

        if min_match_pct > 0:
            from .build_diff import diff_function
            diff = diff_function(self.config, self.unit_name, func_name)
            new_pct = diff.func_match_pct or 0
            if new_pct < min_match_pct:
                self._restore()
                build_unit(self.config, self.source_path, self.unit_name)
                self.build_ok = False
                return EditResult(False, f"degraded match ({new_pct:.1f}% < {min_match_pct:.1f}%)")

        self.build_ok = True
        self.cleanup()
        return EditResult(True)

    def write_placeholder(self, func_name: str, new_code: str) -> EditResult:
        """Replace a placeholder comment with code, verify build, restore on failure."""
        from .source_editor import replace_placeholder

        if not new_code or not new_code.strip():
            return EditResult(False, "empty code")

        self._backup()

        if not replace_placeholder(self.full_path, func_name, new_code):
            self._restore()
            return EditResult(False, "placeholder not found")

        ok, err = build_unit(self.config, self.source_path, self.unit_name)
        if not ok:
            self._restore()
            self.build_ok = False
            return EditResult(False, err)

        self.build_ok = True
        self.cleanup()
        return EditResult(True)
