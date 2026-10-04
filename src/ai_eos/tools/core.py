"""Core tools: web search, URL fetch, sandboxed Python, sandboxed files, memory and knowledge search."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import os
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ai_eos.tools.base import Tool, ToolContext, ToolError, ToolSpec

MAX_OUTPUT = 12000


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


# --- helpers ---------------------------------------------------------------------


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _default_resolve(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


async def assert_public_url(url: str, ctx: ToolContext) -> None:
    """Block requests to internal addresses (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("only absolute http(s) URLs are allowed")
    resolve = ctx.services.get("resolve", _default_resolve)
    try:
        addresses = await asyncio.to_thread(resolve, parsed.hostname)
    except OSError as exc:
        raise ToolError(f"cannot resolve host {parsed.hostname}") from exc
    for addr in addresses:
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ToolError("requests to internal network addresses are blocked")


def sandbox_path(ctx: ToolContext, relative: str) -> Path:
    root = Path(ctx.settings.tools.sandbox_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / relative.lstrip("/\\")).resolve()
    if not target.is_relative_to(root):
        raise ToolError("path escapes the workspace sandbox")
    return target


# --- web ---------------------------------------------------------------------------


class WebSearchTool(Tool):
    spec = ToolSpec(
        "web_search",
        "Search the public web. Returns titles, URLs and snippets.",
        _obj({"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
        timeout=40,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        cfg = ctx.settings.tools.web_search
        n = max(1, min(int(args.get("max_results", 6)), 10))
        if cfg.provider == "none":
            raise ToolError("web search is disabled in configuration")
        if cfg.provider == "tavily":
            key = await ctx.secrets.get(cfg.api_key_env)
            if not key:
                raise ToolError(f"set {cfg.api_key_env} to use Tavily search")
            async with ctx.http() as client:
                r = await client.post(
                    "https://api.tavily.com/search", json={"api_key": key, "query": args["query"], "max_results": n}
                )
                r.raise_for_status()
                results = [(i["title"], i["url"], i.get("content", "")) for i in r.json().get("results", [])]
        else:
            async with ctx.http(headers={"User-Agent": "Mozilla/5.0 (AI-EOS research agent)"}) as client:
                r = await client.post("https://html.duckduckgo.com/html/", data={"q": args["query"]})
                r.raise_for_status()
                results = self._parse_ddg(r.text)[:n]
        if not results:
            return "No results."
        return "\n\n".join(f"[{i + 1}] {t}\n{u}\n{s}" for i, (t, u, s) in enumerate(results))

    @staticmethod
    def _parse_ddg(page: str) -> list[tuple[str, str, str]]:
        out = []
        blocks = re.findall(
            r'(?s)class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=class="result__a"|$)', page
        )
        for href, title, rest in blocks:
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            snippet = re.search(r'(?s)class="result__snippet"[^>]*>(.*?)</a>', rest)
            out.append((html_to_text(title), href, html_to_text(snippet.group(1)) if snippet else ""))
        return out


class FetchUrlTool(Tool):
    spec = ToolSpec(
        "fetch_url",
        "Download a public web page or text document and return its readable text.",
        _obj({"url": {"type": "string"}, "max_chars": {"type": "integer"}}, ["url"]),
        timeout=45,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        url = args["url"]
        limit = min(int(args.get("max_chars", 8000)), MAX_OUTPUT)
        async with ctx.http(headers={"User-Agent": "Mozilla/5.0 (AI-EOS research agent)"}) as client:
            for _ in range(4):  # follow redirects manually so every hop is SSRF-checked
                await assert_public_url(url, ctx)
                r = await client.get(url)
                if r.is_redirect and "location" in r.headers:
                    url = str(r.url.join(r.headers["location"]))
                    continue
                break
            r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "pdf" in ctype:
            from ai_eos.memory.manager import extract_text

            text = extract_text("file.pdf", r.content)
        elif "html" in ctype:
            text = html_to_text(r.text)
        else:
            text = r.text
        return f"URL: {url}\n\n{text[:limit]}"


# --- code & files ----------------------------------------------------------------


def _limit_resources() -> None:  # pragma: no cover - runs in the child process
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
    resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))


class PythonExecTool(Tool):
    spec = ToolSpec(
        "python_exec",
        "Run a Python 3 script for calculation, data analysis or charts. Working directory is the "
        "workspace; save charts/files there. Print what you want to see. No network secrets available.",
        _obj({"code": {"type": "string"}}, ["code"]),
        requires_approval=True,
        timeout=60,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = sandbox_path(ctx, ".")
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=root, delete=False, encoding="utf-8") as f:
            f.write(args["code"])
            script = f.name
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root),
            "MPLBACKEND": "Agg",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                script,
                cwd=root,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), ctx.settings.tools.python_timeout_seconds)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(f"script exceeded {ctx.settings.tools.python_timeout_seconds}s") from None
        finally:
            Path(script).unlink(missing_ok=True)
        text = out.decode(errors="replace")
        if err:
            text += "\n[stderr]\n" + err.decode(errors="replace")
        return f"exit code {proc.returncode}\n{text[-MAX_OUTPUT:]}"


class ReadFileTool(Tool):
    spec = ToolSpec("read_file", "Read a text file from the workspace.", _obj({"path": {"type": "string"}}, ["path"]))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        p = sandbox_path(ctx, args["path"])
        if not p.is_file():
            raise ToolError(f"no such file: {args['path']}")
        return p.read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT]


class WriteFileTool(Tool):
    spec = ToolSpec(
        "write_file",
        "Create or overwrite a text file in the workspace (reports, code, notes).",
        _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        requires_approval=True,
        untrusted_output=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        p = sandbox_path(ctx, args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} characters to {p.relative_to(sandbox_path(ctx, '.'))}"


class ListFilesTool(Tool):
    spec = ToolSpec(
        "list_files",
        "List files in a workspace folder.",
        _obj({"path": {"type": "string"}}, []),
        untrusted_output=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        root = sandbox_path(ctx, ".")
        base = sandbox_path(ctx, args.get("path", "."))
        if not base.is_dir():
            raise ToolError("not a folder")
        files = sorted(str(p.relative_to(root)) for p in base.rglob("*") if p.is_file())[:500]
        return "\n".join(files) or "(empty)"


# --- memory ------------------------------------------------------------------------


class MemorySearchTool(Tool):
    spec = ToolSpec(
        "memory_search",
        "Search the user's long-term memory (facts, preferences, past work).",
        _obj({"query": {"type": "string"}}, ["query"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        items = await ctx.services["memory"].recall(ctx.user_id, args["query"])
        return "\n".join(f"- [{i.kind.value}] {i.content} (score {i.score})" for i in items) or "No memories found."


class KnowledgeSearchTool(Tool):
    spec = ToolSpec(
        "knowledge_search",
        "Search documents the user uploaded to the knowledge base.",
        _obj({"query": {"type": "string"}, "top_k": {"type": "integer"}}, ["query"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        hits = await ctx.services["memory"].search_knowledge(ctx.user_id, args["query"], args.get("top_k"))
        if not hits:
            return "No matching documents."
        return "\n\n".join(f"[{h['title']} #{h['chunk']} score={h['score']}]\n{h['text']}" for h in hits)


CORE_TOOLS: list[type[Tool]] = [
    WebSearchTool,
    FetchUrlTool,
    PythonExecTool,
    ReadFileTool,
    WriteFileTool,
    ListFilesTool,
    MemorySearchTool,
    KnowledgeSearchTool,
]
