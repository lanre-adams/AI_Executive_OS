"""Prompt-injection defences and rate limiting.

Prompt injection cannot be fully solved with pattern matching. This module is one layer:
  1. Input screening flags common injection phrasing in user input and tool output.
  2. Untrusted content (tool results, documents, web pages) is wrapped in delimiters and
     the agents' system prompts tell them to treat it as data, never as instructions.
  3. Tools that act on the outside world (send email, create issues/events) require human
     approval regardless of what the model decides (see tools.base.ToolExecutor).
  4. Filesystem and code tools are sandboxed to one folder with time limits.
Layers 3 and 4 are what actually bound the damage; 1 and 2 reduce how often it is attempted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ai_eos.infrastructure.kv import KeyValueStore

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,30}"
            r"\b(instructions?|prompts?|rules|directions)\b",
            re.I | re.S,
        ),
    ),
    (
        "role_hijack",
        re.compile(r"\byou are now\b|\bact as (?:an? )?(?:unrestricted|jailbroken|dan)\b|\bdeveloper mode\b", re.I),
    ),
    (
        "system_prompt_exfil",
        re.compile(
            r"\b(reveal|print|show|repeat|output)\b.{0,30}\b(system prompt|hidden instructions|your instructions)\b",
            re.I | re.S,
        ),
    ),
    (
        "fake_delimiter",
        re.compile(r"<\s*/?\s*(system|untrusted|instructions?)\s*>|\[\s*system\s*\]|###\s*system", re.I),
    ),
    (
        "secret_exfil",
        re.compile(
            r"\b(send|email|post|upload|exfiltrate|forward)\b.{0,60}\b(api[_ ]?keys?|passwords?|tokens?|secrets?|"
            r"credentials|\.env)\b",
            re.I | re.S,
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(call|invoke|run|execute|use)\b.{0,20}\b(the )?(tool|function)\b.{0,40}\bwithout\b.{0,20}"
            r"\b(asking|approval|confirm)",
            re.I | re.S,
        ),
    ),
]
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")


@dataclass
class ScreenResult:
    flags: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def normalise(text: str) -> str:
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))


def screen(text: str) -> ScreenResult:
    clean = normalise(text)
    flags = [name for name, pat in _PATTERNS if pat.search(clean)]
    if clean != text:
        flags.append("hidden_characters")
    return ScreenResult(flags=flags, text=clean)


def wrap_untrusted(source: str, content: str, limit: int = 12000) -> str:
    """Wrap external content so the model can tell data from instructions."""
    body = normalise(content)
    body = re.sub(r"<\s*/?\s*untrusted[^>]*>", "[removed-tag]", body, flags=re.I)
    result = screen(body)  # screen the full text before truncating
    if len(body) > limit:
        body = body[:limit] + f"\n...[truncated {len(body) - limit} chars]"
    note = f' warning="possible injection: {", ".join(result.flags)}"' if result.suspicious else ""
    return f'<untrusted source="{source}"{note}>\n{body}\n</untrusted>'


class RateLimiter:
    def __init__(self, kv: KeyValueStore, per_minute: int) -> None:
        self.kv, self.per_minute = kv, per_minute

    async def allow(self, identity: str) -> tuple[bool, int]:
        count = await self.kv.incr_window(f"rl:{identity}", 60)
        return count <= self.per_minute, max(0, self.per_minute - count)
