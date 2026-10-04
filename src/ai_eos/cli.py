"""`eos` command-line interface.

Client commands talk to a running server over HTTP (local or remote).
Admin commands (serve, create-user, google-auth, gen-secrets) run locally.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

app = typer.Typer(help="AI-EOS: talk to your Chief of Staff from the terminal.", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect and edit long-term memory.")
app.add_typer(memory_app, name="memory")
console = Console()
CRED_FILE = Path(os.environ.get("EOS_CLI_CREDENTIALS", Path.home() / ".eos" / "credentials.json"))


# ---------------------------------------------------------------- client plumbing
def _load_creds() -> dict[str, str]:
    if CRED_FILE.is_file():
        return json.loads(CRED_FILE.read_text())
    return {}


def _save_creds(data: dict[str, str]) -> None:
    CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    CRED_FILE.write_text(json.dumps(data))
    CRED_FILE.chmod(0o600)


def _client() -> httpx.Client:
    creds = _load_creds()
    server = os.environ.get("EOS_SERVER", creds.get("server", "http://localhost:8000"))
    key = os.environ.get("EOS_API_KEY", creds.get("api_key", ""))
    token = creds.get("token", "")
    headers = {"X-API-Key": key} if key else ({"Authorization": f"Bearer {token}"} if token else {})
    return httpx.Client(base_url=server, headers=headers, timeout=900)


def _call(method: str, path: str, **kwargs: Any) -> Any:
    with _client() as c:
        try:
            r = c.request(method, "/api" + path, **kwargs)
        except httpx.ConnectError:
            console.print(f"[red]Cannot reach the server at {c.base_url}. Is it running? (`eos serve`)[/red]")
            raise typer.Exit(2) from None
    if r.status_code == 401:
        console.print("[red]Not signed in or session expired. Run `eos login`.[/red]")
        raise typer.Exit(1)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        console.print(f"[red]Error {r.status_code}: {detail}[/red]")
        raise typer.Exit(1)
    return None if r.status_code == 204 else r.json()


# ---------------------------------------------------------------- client commands
@app.command()
def login(
    email: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True),
    server: str = typer.Option("http://localhost:8000", help="Server URL"),
) -> None:
    """Sign in and store a session token (or use EOS_API_KEY instead)."""
    _save_creds({"server": server})
    data = _call("POST", "/auth/login", json={"email": email, "password": password})
    _save_creds({"server": server, "token": data["access_token"]})
    console.print(f"[green]Signed in as {data['user']['email']} ({data['user']['role']}).[/green]")


def _show_outcome(data: dict[str, Any]) -> None:
    console.print(Markdown(data["answer"] or "_(no answer)_"))
    agents = ", ".join(r["agent"] for r in data.get("results", []))
    console.print(f"[dim]{data['status']} · task {data['task_id']}{' · ' + agents if agents else ''}[/dim]")


@app.command()
def chat(
    message: str = typer.Argument(None, help="One-shot request. Omit for an interactive session."),
    provider: str = typer.Option(None, "--model", "-m", help="Provider name from settings.yaml"),
) -> None:
    """Talk to the Chief of Staff."""
    conv: str | None = None
    if message:
        _show_outcome(_call("POST", "/chat", json={"message": message, "provider": provider}))
        return
    console.print("[bold]Chief of Staff[/bold] - type your request; 'exit' to quit, 'new' for a fresh conversation.")
    while True:
        try:
            text = console.input("[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("exit", "quit"):
            break
        if text == "new":
            conv = None
            continue
        if not text:
            continue
        with console.status("The team is working…"):
            data = _call("POST", "/chat", json={"message": text, "conversation_id": conv, "provider": provider})
        conv = data["conversation_id"]
        _show_outcome(data)


@app.command()
def tasks(status: str = typer.Option(None, help="Filter by status")) -> None:
    """List recent tasks."""
    rows = _call("GET", "/tasks" + (f"?status={status}" if status else ""))
    t = Table("id", "status", "priority", "request", "s")
    for r in rows[:50]:
        t.add_row(r["id"][:8], r["status"], r["priority"], escape(r["request"][:70]), f"{r['duration_ms'] / 1000:.1f}")
    console.print(t)


@app.command()
def task(task_id: str) -> None:
    """Show a task's plan, agent outputs and final answer."""
    if len(task_id) < 32:
        matches = [r for r in _call("GET", "/tasks") if r["id"].startswith(task_id)]
        if len(matches) != 1:
            console.print("[red]Task id not found or ambiguous.[/red]")
            raise typer.Exit(1)
        task_id = matches[0]["id"]
    data = _call("GET", f"/tasks/{task_id}")
    console.print(
        f"[bold]{escape(data['intent'] or data['request'])}[/bold]  [dim]{data['status']} · {data['priority']}[/dim]"
    )
    for s in data["subtasks"]:
        console.rule(f"{s['agent']} · score {s['score']} · revisions {s['revisions']}")
        console.print(Markdown(s["output"] or "_no output_"))
    console.rule("Final answer")
    console.print(Markdown(data["answer"]))


@app.command()
def approvals(all_: bool = typer.Option(False, "--all", help="Include decided approvals")) -> None:
    """List actions waiting for your approval."""
    rows = _call("GET", "/approvals" + ("" if all_ else "?status=pending"))
    if not rows:
        console.print("Nothing waiting for approval.")
    for a in rows:
        console.print(
            f"[bold]{a['id']}[/bold] {escape('[' + a['status'] + ']')} {a['agent']} → {a['tool']}\n"
            f"  {escape(a['summary'])}\n"
        )


@app.command()
def approve(approval_id: str) -> None:
    """Approve and run a queued action."""
    r = _call("POST", f"/approvals/{approval_id}", json={"approve": True})
    console.print(f"{r['status']}: {escape(r.get('result', ''))}")


@app.command()
def reject(approval_id: str) -> None:
    """Reject a queued action."""
    console.print(_call("POST", f"/approvals/{approval_id}", json={"approve": False})["status"])


@app.command()
def agents() -> None:
    """List agents and their tools."""
    t = Table("key", "title", "model", "tools")
    for a in _call("GET", "/agents"):
        t.add_row(
            a["key"], a["title"], f"{a['provider']}{'/' + a['model'] if a['model'] else ''}", ", ".join(a["tools"])
        )
    console.print(t)


@app.command("run-agent")
def run_agent(agent: str, instruction: str) -> None:
    """Give an instruction directly to one specialist (bypasses the Chief of Staff)."""
    r = _call("POST", f"/agents/{agent}/run", json={"instruction": instruction})
    console.print(Markdown(r["output"]) if r["ok"] else f"[red]{r['error']}[/red]")


@app.command()
def ingest(path: Path, title: str = typer.Option(None)) -> None:
    """Add a document to the knowledge base."""
    with path.open("rb") as fh:
        r = _call("POST", "/knowledge", files={"file": (path.name, fh)}, data={"title": title} if title else {})
    console.print(f"[green]Indexed {r['title']} ({r['chunks']} chunks).[/green]")


@memory_app.command("list")
def memory_list(kind: str = typer.Option(None)) -> None:
    for m in _call("GET", "/memory" + (f"?kind={kind}" if kind else "")):
        console.print(f"[dim]{m['id'][:8]}[/dim] {escape('[' + m['kind'] + ']')} {escape(m['content'])}")


@memory_app.command("search")
def memory_search(query: str) -> None:
    for m in _call("GET", "/memory/search", params={"q": query}):
        console.print(f"{m['score']:.2f} {escape('[' + m['kind'] + ']')} {escape(m['content'])}")


@memory_app.command("add")
def memory_add(content: str, kind: str = typer.Option("fact")) -> None:
    _call("POST", "/memory", json={"kind": kind, "content": content})
    console.print("[green]Saved.[/green]")


@memory_app.command("forget")
def memory_forget(memory_id: str) -> None:
    _call("DELETE", f"/memory/{memory_id}")
    console.print("[green]Deleted.[/green]")


@app.command()
def health() -> None:
    """Check server readiness (database, cache, vector store)."""
    with _client() as c:
        r = c.get("/health/ready")
    console.print_json(r.text)
    raise typer.Exit(0 if r.status_code == 200 else 1)


# ---------------------------------------------------------------- local admin commands
@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:  # noqa: S104
    """Run the API server and dashboard."""
    import uvicorn

    uvicorn.run("ai_eos.api.app:create_app", factory=True, host=host, port=port, reload=reload, proxy_headers=True)


@app.command("create-user")
def create_user(
    email: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True),
    role: str = typer.Option("admin"),
) -> None:
    """Create a user directly in the database (use on the server)."""
    from ai_eos.config import get_settings
    from ai_eos.container import build_container
    from ai_eos.security.auth import hash_password, password_problems

    problems = password_problems(password)
    if problems:
        console.print(f"[red]Password needs {', '.join(problems)}.[/red]")
        raise typer.Exit(1)

    async def go() -> None:
        c = build_container(get_settings())
        await c.db.create_all()
        if await c.users.get_by_email(email):
            console.print("[red]A user with that email already exists.[/red]")
            raise typer.Exit(1)
        await c.users.create(email, hash_password(password), role)
        await c.audit.log("user.created_cli", None, email, {"role": role})
        await c.shutdown()

    asyncio.run(go())
    console.print(f"[green]Created {role} {email}.[/green]")


@app.command("gen-secrets")
def gen_secrets() -> None:
    """Print fresh values for EOS_JWT_SECRET and EOS_ENCRYPTION_KEY."""
    from cryptography.fernet import Fernet

    console.print(f"EOS_JWT_SECRET={secrets.token_urlsafe(48)}")
    console.print(f"EOS_ENCRYPTION_KEY={Fernet.generate_key().decode()}")


@app.command("google-auth")
def google_auth(
    client_id: str = typer.Option(..., envvar="GOOGLE_OAUTH_CLIENT_ID", prompt=True),
    client_secret: str = typer.Option(..., envvar="GOOGLE_OAUTH_CLIENT_SECRET", prompt=True, hide_input=True),
) -> None:
    """Get a Google refresh token for Gmail/Calendar/Drive (OAuth desktop-app flow)."""
    import http.server
    import threading
    import webbrowser

    from ai_eos.tools.google import SCOPES, TOKEN_URL

    port = 8765
    redirect = f"http://127.0.0.1:{port}/"
    state = secrets.token_urlsafe(16)
    result: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            if q.get("state", [""])[0] == state and "code" in q:
                result["code"] = q["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"AI-EOS: authorisation received. You can close this tab.")

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
    )
    console.print(f"Opening your browser. If it doesn't open, visit:\n{url}")
    webbrowser.open(url)
    with console.status("Waiting for you to approve access in the browser…"):
        for _ in range(600):
            if "code" in result:
                break
            import time

            time.sleep(0.5)
    server.server_close()
    if "code" not in result:
        console.print("[red]Timed out waiting for authorisation.[/red]")
        raise typer.Exit(1)
    r = httpx.post(
        TOKEN_URL,
        data={
            "code": result["code"],
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect,
            "grant_type": "authorization_code",
        },
    )
    if r.status_code >= 400:
        console.print(f"[red]Token exchange failed: {r.text}[/red]")
        raise typer.Exit(1)
    refresh = r.json().get("refresh_token")
    console.print("[green]Success.[/green] Add this to your .env (or save it in Settings → Integration secrets):\n")
    console.print(f"GOOGLE_OAUTH_REFRESH_TOKEN={refresh}", soft_wrap=True)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
