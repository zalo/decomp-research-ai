"""Minimal AI prompt generation and API calls for decompilation fixes."""

import re
import os
import json
from typing import Optional
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import URLError

from .config import Config
from .prompts import INITIAL_DECOMPILE, LOGIC_FIX, REGALLOC_FIX, SYNTAX_FIX


@dataclass
class AIResult:
    c_code: Optional[str]
    model: str
    prompt_tokens: int
    response_tokens: int
    success: bool


def _extract_code_block(text: str) -> Optional[str]:
    """Extract C code from a response, handling markdown code blocks.

    Returns None if no valid C code block is found — never returns
    raw reasoning text that could corrupt source files.
    """
    # Try ```c ... ``` block first
    match = re.search(r'```c\n(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if _looks_like_c(code):
            return code

    # Try generic ``` ... ``` block
    match = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if _looks_like_c(code):
            return code

    # If the entire response looks like a C function, use it
    stripped = text.strip()
    if _looks_like_c(stripped):
        return stripped

    # No valid C code found — DO NOT return raw text
    return None


def _looks_like_c(text: str) -> bool:
    """Heuristic: does this look like C code, not prose?"""
    if not text:
        return False
    # Must contain braces (function body)
    if '{' not in text or '}' not in text:
        return False
    # Must not be predominantly prose (more than 50% of lines start with letters forming sentences)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    prose_lines = sum(1 for l in lines if re.match(r'^[A-Z][a-z].*\s[a-z]', l) and '{' not in l and ';' not in l and '#' not in l)
    if prose_lines > len(lines) * 0.3:
        return False
    return True


def _call_ai(messages: list, model: str, config: Config,
             max_tokens: int = 4096, system: str = "") -> AIResult:
    """Call Anthropic Messages API with multi-turn conversation. Returns AIResult."""
    api_key = config.api_key
    if not api_key:
        print(f"    [AI] No API key configured")
        return AIResult(None, model, 0, 0, False)

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        body["system"] = system

    payload = json.dumps(body).encode("utf-8")
    total_chars = sum(len(m.get("content", "")) for m in messages)

    req = Request(
        config.api_base_url + "/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        print(f"    [AI] Calling {model} ({len(messages)} messages, {total_chars} chars)")
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        ptok = usage.get("input_tokens", 0)
        ctok = usage.get("output_tokens", 0)

        print(f"    [AI] Response: {len(text)} chars, {ptok}+{ctok} tokens")

        c_code = _extract_code_block(text)
        if c_code is None:
            print(f"    [AI] No valid C code found. First 200 chars:")
            print(f"    [AI] {repr(text[:200])}")
            return AIResult(None, model, ptok, ctok, False)

        print(f"    [AI] Extracted {len(c_code)} chars of C code")
        return AIResult(
            c_code=c_code,
            model=model,
            prompt_tokens=ptok,
            response_tokens=ctok,
            success=True,
        )
    except Exception as e:
        print(f"    [AI] Error: {e}")
        return AIResult(None, model, 0, 0, False)


class ConversationAgent:
    """Multi-turn AI agent with tool use that accumulates context across retries."""

    def __init__(self, config: Config, system_prompt: str, model: str = None,
                 tool_executor=None):
        self.config = config
        self.model = model or config.initial_model
        self.system = system_prompt
        self.messages = []
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.tool_executor = tool_executor

    def run(self, user_message: str, max_tool_rounds: int = 20) -> AIResult:
        """Send a message and let the agent use tools until it produces C code.

        Tool calls and thinking don't count as attempts — only compile_and_diff does.
        Bails out if 5 consecutive compile attempts show no improvement.
        """
        from .tools import TOOL_DEFINITIONS

        self.messages.append({"role": "user", "content": user_message})
        stale_compiles = 0
        last_best = getattr(self.tool_executor, 'best_match_pct', 0) if self.tool_executor else 0

        for round_num in range(max_tool_rounds):
            result = self._call_with_tools()

            if result is None:
                return AIResult(None, self.model, 0, 0, False)

            # If we got a final text response with C code, return it
            if result.c_code and not result._has_tool_use:
                return result

            # If the model wants to use tools, execute them
            if result._has_tool_use:
                self._execute_tool_calls(result._raw_content)

                # Check if compile_and_diff was called and track staleness
                for block in result._raw_content:
                    if block.get("type") == "tool_use" and block.get("name") == "compile_and_diff":
                        current_best = getattr(self.tool_executor, 'best_match_pct', 0) if self.tool_executor else 0
                        if current_best > last_best:
                            stale_compiles = 0
                            last_best = current_best
                        else:
                            stale_compiles += 1

                        if stale_compiles >= 5:
                            print(f"    [AI] Bailing out: 5 compiles with no improvement")
                            return AIResult(None, self.model, self.total_prompt_tokens,
                                            self.total_response_tokens, False)

                        if current_best == 100.0:
                            print(f"    [AI] PERFECT MATCH achieved!")
                            return AIResult(None, self.model, self.total_prompt_tokens,
                                            self.total_response_tokens, True)
                continue

            # Model responded but no tool use and no code — it's done
            return result

        return AIResult(None, self.model, self.total_prompt_tokens,
                        self.total_response_tokens, False)

    def _call_with_tools(self):
        """Make an API call with tool definitions."""
        from .tools import TOOL_DEFINITIONS

        api_key = self.config.api_key
        if not api_key:
            return None

        body = {
            "model": self.model,
            "max_tokens": 4096,
            "system": self.system,
            "messages": self.messages,
            "tools": TOOL_DEFINITIONS,
        }

        payload = json.dumps(body).encode("utf-8")
        total_chars = sum(len(str(m.get("content", ""))) for m in self.messages)

        req = Request(
            self.config.api_base_url + "/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

        try:
            print(f"    [AI] Round {len(self.messages)//2 + 1}: {len(self.messages)} msgs, {total_chars} chars")
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            usage = data.get("usage", {})
            ptok = usage.get("input_tokens", 0)
            ctok = usage.get("output_tokens", 0)
            self.total_prompt_tokens += ptok
            self.total_response_tokens += ctok

            content = data.get("content", [])
            stop_reason = data.get("stop_reason", "")

            # Check for tool use
            has_tool_use = any(b.get("type") == "tool_use" for b in content)

            # Extract text
            text = ""
            for block in content:
                if block.get("type") == "text":
                    text += block.get("text", "")

            # Log tool calls
            for block in content:
                if block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    print(f"    [AI] → tool: {name}")

            c_code = _extract_code_block(text) if text else None

            result = AIResult(
                c_code=c_code,
                model=self.model,
                prompt_tokens=ptok,
                response_tokens=ctok,
                success=True,
            )
            result._has_tool_use = has_tool_use
            result._raw_content = content

            # Add assistant message to history
            self.messages.append({"role": "assistant", "content": content})

            if text and not has_tool_use:
                print(f"    [AI] Response: {len(text)} chars, code={'yes' if c_code else 'no'}")

            return result

        except Exception as e:
            print(f"    [AI] Error: {e}")
            return None

    def _execute_tool_calls(self, content_blocks):
        """Execute tool calls and add results to message history."""
        tool_results = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                inp = block.get("input", {})

                if self.tool_executor:
                    result_text = self.tool_executor.execute(name, inp)
                    # Truncate very long results
                    if len(result_text) > 3000:
                        result_text = result_text[:3000] + "\n... (truncated)"
                    print(f"    [AI]   {name} → {len(result_text)} chars")
                else:
                    result_text = "Tool executor not available."

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                })

        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})

    # Keep simple ask() for backward compatibility
    def ask(self, user_message: str) -> AIResult:
        """Simple single-turn ask without tools."""
        self.messages.append({"role": "user", "content": user_message})
        result = _call_ai(self.messages, self.model, self.config, system=self.system)
        if result.success and result.c_code:
            self.messages.append({"role": "assistant", "content": f"```c\n{result.c_code}\n```"})
        elif result.prompt_tokens > 0:
            self.messages.append({"role": "assistant", "content": "(no valid C code produced)"})
        self.total_prompt_tokens += result.prompt_tokens
        self.total_response_tokens += result.response_tokens
        return result


def create_decompile_agent(config: Config, target_asm: str,
                           nearby_c: str = "", includes: str = "") -> ConversationAgent:
    """Create a conversation agent for decompiling a function from assembly."""
    agent = ConversationAgent(config, MWCC_CONTEXT)
    # Prime with initial context as first user message
    agent.messages.append({"role": "user", "content": INITIAL_DECOMPILE.format(
        asm=target_asm[:4000],
        signature="(see assembly)",
        nearby_c=nearby_c[:1000] if nearby_c else "(none available)",
        includes=includes[:500] if includes else "(standard melee includes)",
    )})
    # Remove the primed message — agent.ask() will re-add context
    agent.messages.pop()
    return agent


def create_logic_agent(config: Config, current_c: str, target_asm: str,
                       target_size: int, compiled_size: int,
                       diff_details: str, nearby_c: str = "") -> ConversationAgent:
    """Create a conversation agent for fixing logic/size mismatches."""
    agent = ConversationAgent(config, MWCC_CONTEXT)
    return agent


def create_regalloc_agent(config: Config) -> ConversationAgent:
    """Create a conversation agent for fixing register allocation."""
    return ConversationAgent(config, MWCC_CONTEXT)


# Convenience wrappers for single-shot calls (used by simpler handlers)
def initial_decompile(config: Config, target_asm: str, signature: str = "",
                      nearby_c: str = "", includes: str = "",
                      model: Optional[str] = None) -> AIResult:
    from .prompts import MWCC_CONTEXT as _ctx
    prompt = INITIAL_DECOMPILE.format(
        asm=target_asm[:4000],
        signature=signature or "(unknown)",
        nearby_c=nearby_c[:1000] if nearby_c else "(none available)",
        includes=includes[:500] if includes else "(standard melee includes)",
    )
    return _call_ai([{"role": "user", "content": prompt}],
                    model or config.initial_model, config, system=_ctx)


def fix_logic(config: Config, current_c: str, target_asm: str,
              target_size: int, compiled_size: int,
              diff_details: str, nearby_c: str = "",
              model: Optional[str] = None) -> AIResult:
    from .prompts import MWCC_CONTEXT as _ctx
    prompt = LOGIC_FIX.format(
        asm=target_asm[:3000],
        current_c=current_c,
        target_size=target_size,
        compiled_size=compiled_size,
        delta=compiled_size - target_size,
        diff_details=diff_details[:1000],
        nearby_c=nearby_c[:500] if nearby_c else "(none available)",
    )
    return _call_ai([{"role": "user", "content": prompt}],
                    model or config.cheap_model, config, system=_ctx)


def fix_regalloc(config: Config, current_c: str, diff_details: str,
                 model: Optional[str] = None) -> AIResult:
    from .prompts import MWCC_CONTEXT as _ctx
    prompt = REGALLOC_FIX.format(
        current_c=current_c,
        diff_details=diff_details[:1500],
    )
    return _call_ai([{"role": "user", "content": prompt}],
                    model or config.cheap_model, config, system=_ctx)


def fix_syntax(config: Config, current_c: str, error: str,
               model: Optional[str] = None) -> AIResult:
    from .prompts import MWCC_CONTEXT as _ctx
    prompt = SYNTAX_FIX.format(
        current_c=current_c,
        error=error[:500],
    )
    return _call_ai([{"role": "user", "content": prompt}],
                    model or config.cheap_model, config, system=_ctx)
