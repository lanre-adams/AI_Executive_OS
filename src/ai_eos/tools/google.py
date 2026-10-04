"""Google Workspace tools: Gmail, Calendar, Drive.

Authentication uses an OAuth 2.0 user token. Provide either a short-lived access token
(GOOGLE_OAUTH_ACCESS_TOKEN) or, better, a refresh token plus client id/secret so the
system can mint access tokens itself. `eos google-auth` walks you through getting one.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Any

from ai_eos.tools.base import Tool, ToolContext, ToolError, ToolSpec
from ai_eos.tools.core import _obj

GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
CAL = "https://www.googleapis.com/calendar/v3/calendars"
DRIVE = "https://www.googleapis.com/drive/v3/files"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105 - public endpoint, not a secret
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
]


async def google_token(ctx: ToolContext) -> str:
    cfg = ctx.settings.tools.google
    cached = await ctx.kv.get("google:access_token")
    if cached:
        return cached
    refresh = await ctx.secrets.get(cfg.refresh_token_env)
    client_id = await ctx.secrets.get(cfg.client_id_env)
    client_secret = await ctx.secrets.get(cfg.client_secret_env)
    if refresh and client_id and client_secret:
        async with ctx.http() as c:
            r = await c.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
            if r.status_code >= 400:
                raise ToolError(f"Google token refresh failed: {r.text[:200]}")
            data = r.json()
        await ctx.kv.set("google:access_token", data["access_token"], max(60, int(data.get("expires_in", 3600)) - 120))
        return data["access_token"]
    token = await ctx.secrets.get(cfg.access_token_env)
    if token:
        return token
    raise ToolError("Google Workspace is not connected. Run `eos google-auth` (see docs/guides/integrations.md).")


class _Google(Tool):
    async def _client(self, ctx: ToolContext):  # noqa: ANN202
        return ctx.http(headers={"Authorization": f"Bearer {await google_token(ctx)}"})


def _header(headers: list[dict[str, str]], name: str) -> str:
    return next((h["value"] for h in headers if h["name"].lower() == name.lower()), "")


def _plain_body(payload: dict[str, Any]) -> str:
    if payload.get("mimeType", "").startswith("text/plain") and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"] + "==").decode("utf-8", errors="replace")
    for part in payload.get("parts", []) or []:
        text = _plain_body(part)
        if text:
            return text
    if payload.get("mimeType", "").startswith("text/html") and payload.get("body", {}).get("data"):
        from ai_eos.tools.core import html_to_text

        return html_to_text(base64.urlsafe_b64decode(payload["body"]["data"] + "==").decode("utf-8", "replace"))
    return ""


def _raw_email(to: str, subject: str, body: str, cc: str = "", reply_to_id: str = "") -> str:
    msg = EmailMessage()
    msg["To"], msg["Subject"] = to, subject
    if cc:
        msg["Cc"] = cc
    if reply_to_id:
        msg["In-Reply-To"] = msg["References"] = reply_to_id
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


_EMAIL_ARGS = {
    "to": {"type": "string"},
    "subject": {"type": "string"},
    "body": {"type": "string"},
    "cc": {"type": "string"},
    "thread_id": {"type": "string"},
}


class GmailSearch(_Google):
    spec = ToolSpec(
        "gmail_search",
        "Search Gmail with Gmail query syntax (e.g. 'is:unread newer_than:2d').",
        _obj({"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        async with await self._client(ctx) as c:
            r = await c.get(
                f"{GMAIL}/messages",
                params={"q": args["query"], "maxResults": min(int(args.get("max_results", 15)), 50)},
            )
            r.raise_for_status()
            lines = []
            for m in r.json().get("messages", []):
                d = await c.get(
                    f"{GMAIL}/messages/{m['id']}",
                    params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
                )
                d.raise_for_status()
                msg = d.json()
                h = msg.get("payload", {}).get("headers", [])
                unread = "UNREAD " if "UNREAD" in msg.get("labelIds", []) else ""
                lines.append(
                    f"id={m['id']} thread={msg.get('threadId')} {unread}| {_header(h, 'Date')} | "
                    f"{_header(h, 'From')} | {_header(h, 'Subject')} | {msg.get('snippet', '')[:160]}"
                )
        return "\n".join(lines) or "No messages."


class GmailRead(_Google):
    spec = ToolSpec("gmail_read", "Read one email by id.", _obj({"message_id": {"type": "string"}}, ["message_id"]))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        async with await self._client(ctx) as c:
            r = await c.get(f"{GMAIL}/messages/{args['message_id']}", params={"format": "full"})
            r.raise_for_status()
        msg = r.json()
        h = msg.get("payload", {}).get("headers", [])
        return (
            f"From: {_header(h, 'From')}\nTo: {_header(h, 'To')}\nDate: {_header(h, 'Date')}\n"
            f"Subject: {_header(h, 'Subject')}\nMessage-ID: {_header(h, 'Message-ID')}\n\n"
            f"{_plain_body(msg.get('payload', {}))[:10000]}"
        )


class GmailDraft(_Google):
    spec = ToolSpec(
        "gmail_draft",
        "Save an email as a Gmail draft (nothing is sent).",
        _obj(_EMAIL_ARGS, ["to", "subject", "body"]),
        untrusted_output=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        message: dict[str, Any] = {"raw": _raw_email(args["to"], args["subject"], args["body"], args.get("cc", ""))}
        if args.get("thread_id"):
            message["threadId"] = args["thread_id"]
        async with await self._client(ctx) as c:
            r = await c.post(f"{GMAIL}/drafts", json={"message": message})
            r.raise_for_status()
        return f"Draft saved (id {r.json().get('id')}) to {args['to']}: {args['subject']}"


class GmailSend(_Google):
    spec = ToolSpec(
        "gmail_send",
        "Send an email from the user's Gmail.",
        _obj(_EMAIL_ARGS, ["to", "subject", "body"]),
        requires_approval=True,
        untrusted_output=False,
    )

    def summarize(self, args: dict[str, Any]) -> str:
        return f'Send email to {args.get("to")}: "{args.get("subject")}"\n\n{str(args.get("body", ""))[:600]}'

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        body: dict[str, Any] = {"raw": _raw_email(args["to"], args["subject"], args["body"], args.get("cc", ""))}
        if args.get("thread_id"):
            body["threadId"] = args["thread_id"]
        async with await self._client(ctx) as c:
            r = await c.post(f"{GMAIL}/messages/send", json=body)
            r.raise_for_status()
        return f"Email sent to {args['to']} (id {r.json().get('id')})"


class CalendarListEvents(_Google):
    spec = ToolSpec(
        "calendar_list_events",
        "List Google Calendar events between two ISO datetimes (default: next 7 days).",
        _obj({"time_min": {"type": "string"}, "time_max": {"type": "string"}, "calendar_id": {"type": "string"}}, []),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        now = datetime.now(UTC)
        params = {
            "timeMin": args.get("time_min") or now.isoformat(),
            "timeMax": args.get("time_max") or (now + timedelta(days=7)).isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": 100,
        }
        async with await self._client(ctx) as c:
            r = await c.get(f"{CAL}/{args.get('calendar_id', 'primary')}/events", params=params)
            r.raise_for_status()
        lines = []
        for e in r.json().get("items", []):
            start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")
            end = e.get("end", {}).get("dateTime") or e.get("end", {}).get("date")
            who = ", ".join(a.get("email", "") for a in e.get("attendees", [])[:8])
            lines.append(f"{start} -> {end} | {e.get('summary', '(no title)')} | {e.get('location', '')} | {who}")
        return "\n".join(lines) or "No events in that window."


class CalendarCreateEvent(_Google):
    spec = ToolSpec(
        "calendar_create_event",
        "Create a Google Calendar event and invite attendees.",
        _obj(
            {
                "summary": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "timezone": {"type": "string"},
                "attendees": {"type": "array"},
                "description": {"type": "string"},
                "location": {"type": "string"},
            },
            ["summary", "start", "end"],
        ),
        requires_approval=True,
        untrusted_output=False,
    )

    def summarize(self, args: dict[str, Any]) -> str:
        who = ", ".join(args.get("attendees", []) or []) or "no attendees"
        return f'Create event "{args.get("summary")}" {args.get("start")} -> {args.get("end")} with {who}'

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        tz = args.get("timezone", "Africa/Lagos")
        body = {
            "summary": args["summary"],
            "description": args.get("description", ""),
            "location": args.get("location", ""),
            "start": {"dateTime": args["start"], "timeZone": tz},
            "end": {"dateTime": args["end"], "timeZone": tz},
            "attendees": [{"email": a} for a in args.get("attendees", []) or []],
        }
        async with await self._client(ctx) as c:
            r = await c.post(f"{CAL}/primary/events", params={"sendUpdates": "all"}, json=body)
            r.raise_for_status()
        return f"Event created: {r.json().get('htmlLink', '')}"


class DriveSearch(_Google):
    spec = ToolSpec(
        "drive_search", "Search Google Drive file names and contents.", _obj({"query": {"type": "string"}}, ["query"])
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        q = args["query"].replace("\\", "\\\\").replace("'", "\\'")
        params = {
            "q": f"fullText contains '{q}' and trashed = false",
            "pageSize": 20,
            "fields": "files(id,name,mimeType,webViewLink,modifiedTime)",
        }
        async with await self._client(ctx) as c:
            r = await c.get(DRIVE, params=params)
            r.raise_for_status()
        return (
            "\n".join(
                f"{f['name']} ({f['mimeType']}, {f.get('modifiedTime', '')}) {f.get('webViewLink', '')}"
                for f in r.json().get("files", [])
            )
            or "No files."
        )


GOOGLE_TOOLS: list[type[Tool]] = [
    GmailSearch,
    GmailRead,
    GmailDraft,
    GmailSend,
    CalendarListEvents,
    CalendarCreateEvent,
    DriveSearch,
]
