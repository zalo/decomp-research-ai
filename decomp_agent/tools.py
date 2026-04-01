"""Tools that the AI agent can call during decompilation.

These are exposed as Anthropic tool-use functions. The agent can call them
to inspect assembly, compile code, check diffs, and get Ghidra pseudocode.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Optional

from .config import Config


# Tool definitions for Anthropic API tool_use format
# Only tools the agent actually needs — context is pre-loaded in the prompt
TOOL_DEFINITIONS = [
    {
        "name": "compile_and_diff",
        "description": "Compile your C code with MWCC and diff against the target binary. Returns match %, size comparison, and your compiled assembly so you can compare instruction-by-instruction against the target. The code must be a complete function definition with return type, name, params, and body.",
        "input_schema": {
            "type": "object",
            "properties": {
                "c_code": {"type": "string", "description": "Complete C function code to compile and test"}
            },
            "required": ["c_code"]
        }
    },
    {
        "name": "side_by_side_diff",
        "description": "Show a side-by-side comparison of target assembly vs your compiled assembly, highlighting exactly which instructions differ. Use this AFTER compile_and_diff to understand what your code generates differently.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "ghidra_decompile",
        "description": "Run the Ghidra decompiler on the current function. Returns Ghidra's C pseudocode which may have different structure than m2c. Useful as a second opinion when stuck.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_struct_field",
        "description": "Search project headers and context files for struct fields by hex offset or name. Also searches the current source file for usage patterns. Example: search '0xF8' or 'xDD4' to find field definitions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Hex offset like '0xF8', field name like 'cur_pos', or partial name like 'xDD4'"},
                "struct_type": {"type": "string", "description": "Struct name like 'Ground' or 'Item' to narrow search", "default": ""}
            },
            "required": ["query"]
        }
    },
]


class ToolExecutor:
    """Executes tool calls from the AI agent."""

    def __init__(self, config: Config, func_name: str, unit_name: str,
                 source_path: str, asm_path: str):
        self.config = config
        self.func_name = func_name
        self.unit_name = unit_name
        self.source_path = source_path
        self.asm_path = asm_path
        self.src_full = config.melee_root / source_path
        self.compile_count = 0
        self.best_match_pct = 0.0  # set by orchestrator before use
        self.best_code = None  # best C code found so far

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "ghidra_decompile":
                return self._ghidra_decompile()
            elif tool_name == "compile_and_diff":
                return self._compile_and_diff(tool_input.get("c_code", ""))
            elif tool_name == "side_by_side_diff":
                return self._side_by_side_diff()
            elif tool_name == "search_struct_field":
                return self._search_struct_field(
                    tool_input.get("query", ""),
                    tool_input.get("struct_type", ""))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {e}"

    def _get_target_assembly(self) -> str:
        from .source_editor import extract_function_asm
        asm_path = Path(self.config.melee_root / self.asm_path)
        if not asm_path.exists():
            unit_rel = self.source_path.removeprefix("src/").replace(".c", ".s")
            asm_path = self.config.asm_root / unit_rel
        asm = extract_function_asm(asm_path, self.func_name)
        return asm if asm else "Assembly not found for this function."

    def _get_current_c_code(self) -> str:
        from .source_editor import extract_function_c
        code = extract_function_c(self.src_full, self.func_name)
        return code if code else "No C code found for this function."

    def _run_m2c(self) -> str:
        from .m2c_runner import decompile_function
        result = decompile_function(self.config, self.func_name,
                                    self.source_path, self.asm_path)
        return result if result else "m2c failed to decompile this function."

    def _compile_and_diff(self, c_code: str) -> str:
        """Compile code in a temp swap, diff, then restore original.

        Only persists the change if it's a PERFECT MATCH or improves match %.
        """
        from .safe_edit import SafeFile
        from .build_diff import diff_function, build_unit
        from .source_editor import replace_function, extract_function_c
        import shutil

        if not c_code.strip():
            return "Error: empty code provided."

        self.compile_count += 1
        src_full = self.config.melee_root / self.source_path
        backup = src_full.with_suffix(".c.agent_bak")

        # Backup original
        shutil.copy2(src_full, backup)

        try:
            # Write the new code
            old_code = extract_function_c(src_full, self.func_name)
            if old_code is None:
                return (f"Error: could not find function `{self.func_name}` in source file. "
                        f"Make sure your code includes the full function signature.")

            # Validate the signature hasn't changed
            old_sig = old_code.split("{")[0].strip()
            new_sig = c_code.split("{")[0].strip() if "{" in c_code else ""
            if old_sig and new_sig and old_sig != new_sig:
                return (f"SIGNATURE MISMATCH — you MUST keep the exact original signature.\n"
                        f"Original: {old_sig}\n"
                        f"Yours:    {new_sig}\n"
                        f"Fix your code to use the original signature and try again.")

            if not replace_function(src_full, self.func_name, c_code):
                return (f"Error: could not replace function.\n"
                        f"Original starts with: {old_code[:100]}...")

            # Compile
            ok, err = build_unit(self.config, self.source_path, self.unit_name)
            if not ok:
                return f"COMPILE ERROR:\n{err}\n\nYour code was:\n```c\n{c_code}\n```"

            # Diff
            diff = diff_function(self.config, self.unit_name, self.func_name)

            pct = diff.func_match_pct or 0
            result = f"Match: {pct:.1f}%\n"
            result += f"Size: target={diff.func_size_target}b compiled={diff.func_size_compiled}b\n"

            # Log to stdout for visibility
            print(f"    [TOOL] compile_and_diff: {pct:.1f}% (target={diff.func_size_target}b compiled={diff.func_size_compiled}b) best={self.best_match_pct:.1f}%")

            if diff.is_matched:
                if backup.exists():
                    backup.unlink()
                self.best_match_pct = 100.0
                self.best_code = c_code
                result += "PERFECT MATCH! This code is correct!"
                return result

            if diff.is_regalloc_only:
                result += "\nLogic is CORRECT! Only register allocation differs.\n"

            # Show compiled disassembly for comparison
            disasm = self._disassemble_compiled()
            if disasm and len(disasm) < 2000:
                result += f"\nYour compiled assembly:\n{disasm}\n"

            # Always restore to original after testing. The orchestrator
            # will apply the final best version at the end.
            new_pct = diff.func_match_pct or 0
            if new_pct > self.best_match_pct:
                self.best_match_pct = new_pct
                self.best_code = c_code  # save the best code for later

            # Restore original source
            shutil.copy2(backup, src_full)
            build_unit(self.config, self.source_path, self.unit_name)

            return result

        except Exception as e:
            # Always restore on error
            if backup.exists():
                shutil.copy2(backup, src_full)
                build_unit(self.config, self.source_path, self.unit_name)
            return f"Error: {e}"
        finally:
            # Clean up backup if it still exists and we're done
            pass  # backup cleaned up on match or kept for restore

    def _get_nearby_functions(self) -> str:
        from .source_editor import get_nearby_matched_c
        from .target_selector import load_report
        report = load_report(self.config)
        for u in report.get("units", []):
            if u["name"] == self.unit_name:
                nearby = get_nearby_matched_c(self.src_full,
                                              u.get("functions", []),
                                              self.func_name, max_count=2,
                                              max_size=300)
                if nearby:
                    return "\n\n".join(f"// {name}\n{code}" for name, code in nearby)
        return "No nearby matched functions found."

    def _get_includes(self) -> str:
        if not self.src_full.exists():
            return "Source file not found."
        lines = self.src_full.read_text().split("\n")
        includes = [l for l in lines[:40] if l.startswith("#include") or l.startswith("#pragma")]
        return "\n".join(includes) if includes else "No includes found."

    def _search_struct_field(self, query: str, struct_type: str = "") -> str:
        """Search ALL project headers for struct fields by offset or name.

        Searches recursively through src/ and extern/ for .h files containing
        the query string. If struct_type is given, tries to find that struct
        and show its fields around the matching offset.
        """
        results = []
        src_dir = self.config.melee_root / "src"
        extern_dir = self.config.melee_root / "extern"

        # Search the SOURCE FILE itself first — existing code shows correct field names
        src_full = self.config.melee_root / self.source_path
        if src_full.exists():
            text = src_full.read_text()
            for line in text.split("\n"):
                if query.lower() in line.lower():
                    stripped = line.strip()
                    if stripped and len(stripped) < 200 and not stripped.startswith("//"):
                        results.append(f"[this file] {stripped}")
                        if len(results) >= 5:
                            break

        # Also try the .ctx file which has ALL resolved types
        ctx_path = self.config.melee_root / "build" / "GALE01" / self.source_path.replace("src/", "src/").replace(".c", ".ctx")

        # Search the context file first — it has everything resolved
        if ctx_path.exists():
            text = ctx_path.read_text()
            if struct_type:
                # Find struct block and search within it
                pattern = rf'(?:typedef\s+)?struct\s+\w*{re.escape(struct_type)}\w*\s*\{{(.*?)\}}'
                for m in re.finditer(pattern, text, re.DOTALL):
                    for line in m.group(0).split("\n"):
                        if query.lower() in line.lower():
                            results.append(f"[ctx] {line.strip()}")
            else:
                for line in text.split("\n"):
                    if query.lower() in line.lower():
                        stripped = line.strip()
                        # Skip noise lines
                        if stripped and not stripped.startswith("//") and len(stripped) < 200:
                            results.append(f"[ctx] {stripped}")
                            if len(results) >= 15:
                                break

        # If ctx didn't find enough, search .h files
        if len(results) < 5:
            search_dirs = [src_dir, extern_dir]
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for hdr in search_dir.rglob("*.h"):
                    try:
                        text = hdr.read_text()
                    except Exception:
                        continue

                    if struct_type:
                        pattern = rf'(?:typedef\s+)?struct\s+\w*{re.escape(struct_type)}\w*\s*\{{(.*?)\}}'
                        for m in re.finditer(pattern, text, re.DOTALL):
                            for line in m.group(0).split("\n"):
                                if query.lower() in line.lower():
                                    rel = hdr.relative_to(self.config.melee_root)
                                    results.append(f"{rel}: {line.strip()}")
                    elif query.lower() in text.lower():
                        for line in text.split("\n"):
                            if query.lower() in line.lower():
                                stripped = line.strip()
                                if stripped and not stripped.startswith("//") and len(stripped) < 200:
                                    rel = hdr.relative_to(self.config.melee_root)
                                    results.append(f"{rel}: {stripped}")
                                    if len(results) >= 20:
                                        break
                    if len(results) >= 20:
                        break

        # If searching a hex offset and found nothing, try computing common base offsets
        if not results and query.startswith("0x"):
            try:
                addr = int(query, 16)
                # Common struct base offsets in Melee
                bases = {
                    0xDD4: "xDD4_itemVar",  # Item variant union
                    0x2340: "mv",  # Fighter motion vars
                    0x914: "x914",  # HitCapsule array
                    0xB0: "cur_pos",  # common position offset
                }
                for base, name in bases.items():
                    if addr > base and addr < base + 0x200:
                        field_off = addr - base
                        results.append(f"[computed] offset {query} = {name} + 0x{field_off:X} (field x{field_off:X} in the {name} union/struct)")
                        # Search for that field pattern in source
                        pattern = f"x{field_off:X}"
                        for line in (self.config.melee_root / self.source_path).read_text().split("\n"):
                            if pattern.lower() in line.lower() and name in line:
                                results.append(f"[this file] {line.strip()}")
                                break
            except ValueError:
                pass

        if results:
            return "\n".join(results[:20])
        return f"No results for '{query}'" + (f" in struct '{struct_type}'" if struct_type else "") + "\nTry searching the source file for similar field access patterns."

    def _ghidra_decompile(self) -> str:
        """Run Ghidra headless decompiler on the function."""
        try:
            from .ghidra_runner import decompile_with_ghidra
            result = decompile_with_ghidra(self.config, self.func_name, self.source_path)
            return result if result else "Ghidra decompilation failed or produced no output."
        except Exception as e:
            return f"Ghidra error: {e}"

    def _side_by_side_diff(self) -> str:
        """Show side-by-side target vs compiled assembly with diff markers."""
        target_asm = self._get_target_assembly()
        compiled_asm = self._disassemble_compiled()

        if not target_asm or "not found" in target_asm.lower():
            return "Target assembly not available."
        if not compiled_asm or "not found" in compiled_asm.lower():
            return "Compiled assembly not available. Run compile_and_diff first."

        # Parse both into instruction lists
        def parse_asm(text):
            instrs = []
            for line in text.split("\n"):
                line = line.strip()
                if not line or line.startswith(".") or line.startswith("#"):
                    continue
                # Extract just the instruction mnemonic + operands
                # Target format: /* ADDR OFFSET  BYTES */  instruction
                m = re.search(r'\*/\s+(.+)$', line)
                if m:
                    instrs.append(m.group(1).strip())
                    continue
                # Compiled format:  addr:  bytes  instruction
                m = re.search(r'[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+(.+)$', line)
                if m:
                    instrs.append(m.group(1).strip())
            return instrs

        target = parse_asm(target_asm)
        compiled = parse_asm(compiled_asm)

        # Simple alignment — show both side by side
        result = f"{'TARGET':>40}  |  {'COMPILED':<40}\n"
        result += "-" * 85 + "\n"

        max_len = max(len(target), len(compiled))
        for i in range(max_len):
            t = target[i] if i < len(target) else ""
            c = compiled[i] if i < len(compiled) else ""
            marker = "  " if t == c else ">>"
            result += f"{t:>40} {marker} {c:<40}\n"

        result += f"\nTarget: {len(target)} instructions ({len(target)*4}b)"
        result += f"\nCompiled: {len(compiled)} instructions ({len(compiled)*4}b)"
        result += f"\nDifferences: {sum(1 for i in range(max_len) if (target[i] if i < len(target) else '') != (compiled[i] if i < len(compiled) else ''))}"

        return result

    def _disassemble_compiled(self) -> str:
        # Find the compiled .o file
        import json
        with open(self.config.objdiff_json_path) as f:
            data = json.load(f)
        base_path = None
        for u in data["units"]:
            if u["name"] == self.unit_name:
                base_path = self.config.melee_root / u["base_path"]
                break
        if not base_path or not base_path.exists():
            return "Compiled object not found."

        result = subprocess.run(
            ["powerpc-eabi-objdump", "-d", str(base_path)],
            capture_output=True, text=True, timeout=10
        )
        # Extract just our function
        lines = result.stdout.split("\n")
        in_func = False
        func_lines = []
        for line in lines:
            if f"<{self.func_name}>" in line:
                in_func = True
            if in_func:
                func_lines.append(line)
                if line.strip() == "" and len(func_lines) > 2:
                    break
                if "blr" in line:
                    func_lines.append("")
                    break
        return "\n".join(func_lines[:60]) if func_lines else "Function not found in disassembly."
