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
    """Call AI API. Auto-detects Anthropic vs OpenRouter from config.api_base_url."""
    api_key = config.api_key
    if not api_key:
        print(f"    [AI] No API key configured")
        return AIResult(None, model, 0, 0, False)

    is_anthropic = "anthropic.com" in config.api_base_url
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)

    if is_anthropic:
        body = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            body["system"] = system
        url = config.api_base_url + "/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    else:
        # OpenRouter (OpenAI-compatible)
        # Convert system prompt to first user message if needed
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for m in messages:
            content = m.get("content", "")
            # Flatten Anthropic tool_result format to text for OpenRouter
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            parts.append(f"[Tool result]: {block.get('content', '')}")
                        elif block.get("type") == "tool_use":
                            parts.append(f"[Called tool {block.get('name', '')}]")
                        elif block.get("type") == "text":
                            parts.append(block.get("text", ""))
                content = "\n".join(parts) if parts else str(content)
            oai_messages.append({"role": m["role"], "content": content})
        body = {"model": model, "max_tokens": max_tokens, "messages": oai_messages}
        url = config.api_base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    payload = json.dumps(body).encode("utf-8")
    req = Request(url, data=payload, headers=headers)

    try:
        print(f"    [AI] Calling {model} ({len(messages)} msgs, {total_chars} chars)")
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Parse response — different formats
        if is_anthropic:
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            usage = data.get("usage", {})
            ptok = usage.get("input_tokens", 0)
            ctok = usage.get("output_tokens", 0)
        else:
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning") or ""
            usage = data.get("usage", {})
            ptok = usage.get("prompt_tokens", 0)
            ctok = usage.get("completion_tokens", 0)

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

        For Anthropic API: uses native tool_use.
        For OpenRouter: falls back to ask() without tools (no tool_use support).
        """
        is_anthropic = "anthropic.com" in self.config.api_base_url

        # Both Anthropic and OpenRouter support tool use (different formats)

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
        """Make an API call with tool definitions. Supports Anthropic and OpenRouter."""
        from .tools import TOOL_DEFINITIONS

        api_key = self.config.api_key
        if not api_key:
            return None

        is_anthropic = "anthropic.com" in self.config.api_base_url
        total_chars = sum(len(str(m.get("content", ""))) for m in self.messages)

        if is_anthropic:
            body = {
                "model": self.model,
                "max_tokens": 4096,
                "system": self.system,
                "messages": self.messages,
                "tools": TOOL_DEFINITIONS,
            }
            url = self.config.api_base_url + "/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            # OpenRouter — convert tools to OpenAI format
            oai_tools = []
            for t in TOOL_DEFINITIONS:
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    }
                })
            # Flatten messages for OpenRouter
            oai_messages = []
            if self.system:
                oai_messages.append({"role": "system", "content": self.system})
            for m in self.messages:
                content = m.get("content", "")
                role = m["role"]
                if isinstance(content, list):
                    # Flatten Anthropic format
                    parts = []
                    tool_calls_out = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                tool_calls_out.append({
                                    "type": "function",
                                    "id": block.get("id", ""),
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input", {})),
                                    }
                                })
                            elif block.get("type") == "tool_result":
                                oai_messages.append({
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": block.get("content", ""),
                                })
                                continue
                    if tool_calls_out:
                        oai_messages.append({"role": "assistant", "content": "\n".join(parts) if parts else None, "tool_calls": tool_calls_out})
                    elif parts:
                        oai_messages.append({"role": role, "content": "\n".join(parts)})
                else:
                    oai_messages.append({"role": role, "content": content})

            body = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": oai_messages,
                "tools": oai_tools,
            }
            url = self.config.api_base_url + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

        payload = json.dumps(body).encode("utf-8")
        req = Request(url, data=payload, headers=headers)

        try:
            print(f"    [AI] Round {len(self.messages)//2 + 1}: {len(self.messages)} msgs, {total_chars} chars")
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if is_anthropic:
                usage = data.get("usage", {})
                ptok = usage.get("input_tokens", 0)
                ctok = usage.get("output_tokens", 0)
                content = data.get("content", [])
                has_tool_use = any(b.get("type") == "tool_use" for b in content)
                text = ""
                for block in content:
                    if block.get("type") == "text":
                        text += block.get("text", "")
                for block in content:
                    if block.get("type") == "tool_use":
                        print(f"    [AI] → tool: {block.get('name', '?')}")
                # Add to history in Anthropic format
                self.messages.append({"role": "assistant", "content": content})
            else:
                # OpenRouter response
                usage = data.get("usage", {})
                ptok = usage.get("prompt_tokens", 0)
                ctok = usage.get("completion_tokens", 0)
                choice = data["choices"][0]
                msg = choice["message"]
                text = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                has_tool_use = len(tool_calls) > 0

                # Convert to Anthropic-style content blocks for internal use
                content = []
                if text:
                    content.append({"type": "text", "text": text})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": parsed_args,
                    })
                    print(f"    [AI] → tool: {func.get('name', '?')}")
                self.messages.append({"role": "assistant", "content": content})

            self.total_prompt_tokens += ptok
            self.total_response_tokens += ctok

            c_code = _extract_code_block(text) if text else None
            result = AIResult(
                c_code=c_code, model=self.model,
                prompt_tokens=ptok, response_tokens=ctok, success=True,
            )
            result._has_tool_use = has_tool_use
            result._raw_content = content

            if text and not has_tool_use:
                print(f"    [AI] Response: {len(text)} chars, code={'yes' if c_code else 'no'}")

            return result

        except Exception as e:
            print(f"    [AI] Error: {e}")
            return None

    def _execute_tool_calls(self, content_blocks):
        """Execute tool calls and add results to message history."""
        is_anthropic = "anthropic.com" in self.config.api_base_url
        tool_results = []

        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                inp = block.get("input", {})

                if self.tool_executor:
                    result_text = self.tool_executor.execute(name, inp)
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
            if is_anthropic:
                self.messages.append({"role": "user", "content": tool_results})
            else:
                # OpenRouter: each tool result is a separate message
                for tr in tool_results:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"],
                    })

    def _run_simple(self, user_message: str, max_rounds: int = 10) -> AIResult:
        """Simple multi-turn loop for models without tool_use support.

        Gives the AI all context upfront, asks for code, compiles it,
        and feeds back the result. No tool calls.
        """
        stale = 0
        last_best = getattr(self.tool_executor, 'best_match_pct', 0) if self.tool_executor else 0

        result = self.ask(user_message)

        for round_num in range(max_rounds):
            if not result.success or not result.c_code:
                print(f"    [AI] Round {round_num+1}: no valid code")
                break

            # Manually compile and diff
            if self.tool_executor:
                diff_result = self.tool_executor.execute("compile_and_diff", {"c_code": result.c_code})
                current_best = self.tool_executor.best_match_pct

                if current_best == 100.0:
                    print(f"    [AI] PERFECT MATCH!")
                    return result

                if current_best > last_best:
                    stale = 0
                    last_best = current_best
                else:
                    stale += 1

                if stale >= 5:
                    print(f"    [AI] Bailing: 5 rounds no improvement")
                    break

                result = self.ask(f"Result:\n{diff_result}\n\nFix the code and try again.")
            else:
                break

        return result

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
    # Escape curly braces in user content to prevent format() KeyError
    safe_c = current_c.replace("{", "{{").replace("}", "}}")
    safe_diff = diff_details.replace("{", "{{").replace("}", "}}")
    prompt = REGALLOC_FIX.format(
        current_c=safe_c,
        diff_details=safe_diff,
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
