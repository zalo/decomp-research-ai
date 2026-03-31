"""Read and write individual functions in .c source files."""

import re
from pathlib import Path
from typing import Optional


def extract_function_asm(asm_path: Path, func_name: str) -> Optional[str]:
    """Extract a single function's assembly from a .s file."""
    if not asm_path.exists():
        return None
    text = asm_path.read_text()
    pattern = rf'(\.fn {re.escape(func_name)},.*?\n.*?\.endfn {re.escape(func_name)})'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def extract_function_c(source_path: Path, func_name: str) -> Optional[str]:
    """Extract a C function DEFINITION (with body) from a source file.

    Skips forward declarations (ending with ;) and finds the definition
    with a { body }.
    """
    if not source_path.exists():
        return None
    text = source_path.read_text()

    # Find function DEFINITION: return_type func_name(params) { body }
    # Must match the function name at the start of a line (after optional type/qualifiers)
    # NOT in the middle of an expression or data table
    pattern = rf'^[^\S\n]*(?:static\s+|inline\s+|extern\s+)*\w[\w\s\*]*\b{re.escape(func_name)}\s*\('
    for match in re.finditer(pattern, text, re.MULTILINE):
        idx = match.start()

        # Find the closing paren
        paren_start = text.find("(", match.end() - 1)
        if paren_start == -1:
            continue
        depth = 1
        pos = paren_start + 1
        while pos < len(text) and depth > 0:
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1

        # After close paren, skip whitespace and check for {
        after_paren = text[pos:pos + 20].lstrip()
        if not after_paren.startswith("{"):
            continue

        # Found a definition! Find the opening brace
        brace_start = text.index("{", pos - 1)

        # Count braces to find the end
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[idx:i + 1]
    return None


def has_placeholder(source_path: Path, func_name: str) -> bool:
    """Check if a function has ONLY a placeholder comment (/// #func_name)
    with no existing function body after it.

    Returns False if the function already has code (even if the placeholder
    comment is still there above it).
    """
    text = source_path.read_text()
    pattern = rf'^/// #{re.escape(func_name)}$'
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return False

    # Check if there's already a function definition (not just a comment)
    if extract_function_c(source_path, func_name) is not None:
        return False  # function body already exists, use replace_function instead

    return True


def replace_placeholder(source_path: Path, func_name: str, c_code: str) -> bool:
    """Replace a /// #func_name placeholder with actual C code.

    Also removes any forward declaration of the same function to avoid
    'object already defined' MWCC errors.
    """
    text = source_path.read_text()

    # Replace the placeholder comment
    pattern = rf'^/// #{re.escape(func_name)}$(?:\r?\n)?'
    new_text, count = re.subn(pattern, c_code + "\n", text, flags=re.MULTILINE)
    if count == 0:
        return False

    # Remove forward declarations that would conflict with the new definition.
    # Match: optional "static" + return type + func_name + (params) + ;
    # But NOT the definition we just inserted (which has { not ;)
    decl_pattern = rf'^[ \t]*(?:static\s+)?[^;\n]*\b{re.escape(func_name)}\s*\([^)]*\)\s*;\s*(?:/\*.*?\*/\s*)?\n?'
    new_text = re.sub(decl_pattern, '', new_text, flags=re.MULTILINE)

    source_path.write_text(new_text)
    return True


def replace_function(source_path: Path, func_name: str, new_code: str) -> bool:
    """Replace an existing function in a source file with new code."""
    old_code = extract_function_c(source_path, func_name)
    if old_code is None:
        return False
    text = source_path.read_text()
    new_text = text.replace(old_code, new_code, 1)
    if new_text == text:
        return False
    source_path.write_text(new_text)
    return True


def get_nearby_matched_c(source_path: Path, report_funcs: list,
                         exclude: str, max_count: int = 2,
                         max_size: int = 200) -> list[tuple[str, str]]:
    """Get small matched functions from the same file for style reference."""
    results = []
    matched = [f for f in report_funcs
               if f.get("fuzzy_match_percent") == 100.0
               and int(f.get("size", "0")) <= max_size
               and f["name"] != exclude]
    matched.sort(key=lambda f: int(f["size"]))

    for f in matched[:max_count]:
        code = extract_function_c(source_path, f["name"])
        if code and len(code) < 500:
            results.append((f["name"], code))
    return results
