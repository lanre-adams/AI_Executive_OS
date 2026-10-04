"""GitHub and Azure DevOps tools (REST APIs)."""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import quote

from ai_eos.tools.base import Tool, ToolContext, ToolError, ToolSpec
from ai_eos.tools.core import _obj

# --- GitHub ------------------------------------------------------------------------


class _GitHub(Tool):
    async def _headers(self, ctx: ToolContext) -> dict[str, str]:
        token = await ctx.secrets.get(ctx.settings.tools.github.token_env)
        if not token:
            raise ToolError(f"GitHub is not connected: set {ctx.settings.tools.github.token_env}")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get(self, ctx: ToolContext, path: str, params: dict[str, Any] | None = None) -> Any:
        async with ctx.http(headers=await self._headers(ctx)) as c:
            r = await c.get(ctx.settings.tools.github.api_url + path, params=params)
            r.raise_for_status()
            return r.json()


_REPO = {"repo": {"type": "string", "description": "owner/name"}}


class GitHubListIssues(_GitHub):
    spec = ToolSpec(
        "github_list_issues",
        "List open issues (and PRs) in a repository.",
        _obj({**_REPO, "state": {"type": "string"}, "labels": {"type": "string"}}, ["repo"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        params = {"state": args.get("state", "open"), "per_page": 30}
        if args.get("labels"):
            params["labels"] = args["labels"]
        items = await self._get(ctx, f"/repos/{args['repo']}/issues", params)
        return (
            "\n".join(
                f"#{i['number']} [{'PR' if 'pull_request' in i else 'issue'}] {i['title']} "
                f"({', '.join(lbl['name'] for lbl in i.get('labels', []))}) {i['html_url']}"
                for i in items
            )
            or "No issues."
        )


class GitHubCreateIssue(_GitHub):
    spec = ToolSpec(
        "github_create_issue",
        "Open a new GitHub issue.",
        _obj(
            {**_REPO, "title": {"type": "string"}, "body": {"type": "string"}, "labels": {"type": "array"}},
            ["repo", "title"],
        ),
        requires_approval=True,
        untrusted_output=False,
    )

    def summarize(self, args: dict[str, Any]) -> str:
        return f'Create issue in {args.get("repo")}: "{args.get("title")}"'

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        body = {"title": args["title"], "body": args.get("body", ""), "labels": args.get("labels", [])}
        async with ctx.http(headers=await self._headers(ctx)) as c:
            r = await c.post(f"{ctx.settings.tools.github.api_url}/repos/{args['repo']}/issues", json=body)
            r.raise_for_status()
            data = r.json()
        return f"Created issue #{data['number']}: {data['html_url']}"


class GitHubSearchCode(_GitHub):
    spec = ToolSpec(
        "github_search_code",
        "Search code on GitHub (e.g. 'repo:owner/name AuthService').",
        _obj({"query": {"type": "string"}}, ["query"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        data = await self._get(ctx, "/search/code", {"q": args["query"], "per_page": 20})
        return "\n".join(f"{i['repository']['full_name']}:{i['path']}" for i in data.get("items", [])) or "No matches."


class GitHubGetFile(_GitHub):
    spec = ToolSpec(
        "github_get_file",
        "Read a file from a GitHub repository.",
        _obj({**_REPO, "path": {"type": "string"}, "ref": {"type": "string"}}, ["repo", "path"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        params = {"ref": args["ref"]} if args.get("ref") else None
        data = await self._get(ctx, f"/repos/{args['repo']}/contents/{quote(args['path'])}", params)
        if isinstance(data, list):
            return "Directory listing:\n" + "\n".join(f"{d['type']}: {d['path']}" for d in data)
        return base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")[:12000]


class GitHubListWorkflowRuns(_GitHub):
    spec = ToolSpec(
        "github_list_workflow_runs",
        "List recent GitHub Actions runs and their status.",
        _obj({**_REPO, "branch": {"type": "string"}}, ["repo"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        params: dict[str, Any] = {"per_page": 15}
        if args.get("branch"):
            params["branch"] = args["branch"]
        data = await self._get(ctx, f"/repos/{args['repo']}/actions/runs", params)
        return (
            "\n".join(
                f"{r['name']} #{r['run_number']} {r['head_branch']} {r['status']}/{r.get('conclusion') or '-'} "
                f"{r['created_at']} {r['html_url']}"
                for r in data.get("workflow_runs", [])
            )
            or "No runs."
        )


# --- Azure DevOps --------------------------------------------------------------------


class _AzDO(Tool):
    async def _client_args(self, ctx: ToolContext) -> tuple[str, dict[str, str]]:
        cfg = ctx.settings.tools.azure_devops
        org = await ctx.secrets.get(cfg.organization_env)
        pat = await ctx.secrets.get(cfg.token_env)
        if not org or not pat:
            raise ToolError(f"Azure DevOps is not connected: set {cfg.organization_env} and {cfg.token_env}")
        auth = base64.b64encode(f":{pat}".encode()).decode()
        return f"https://dev.azure.com/{org}", {"Authorization": f"Basic {auth}"}


_PROJECT = {"project": {"type": "string"}}


class AzdoListWorkItems(_AzDO):
    spec = ToolSpec(
        "azdo_list_work_items",
        "List active work items in an Azure DevOps project (optionally by type).",
        _obj({**_PROJECT, "work_item_type": {"type": "string"}, "top": {"type": "integer"}}, ["project"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        base, headers = await self._client_args(ctx)
        project = args["project"].replace("'", "''")
        # WIQL, not SQL; single quotes in user input are escaped above.
        wiql = (
            f"SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = '{project}' "  # noqa: S608
            "AND [System.State] NOT IN ('Closed','Done','Removed')"
        )
        if args.get("work_item_type"):
            wiql += f" AND [System.WorkItemType] = '{args['work_item_type'].replace(chr(39), '')}'"
        wiql += " ORDER BY [System.ChangedDate] DESC"
        async with ctx.http(headers=headers) as c:
            r = await c.post(
                f"{base}/{quote(args['project'])}/_apis/wit/wiql?api-version=7.1&$top={args.get('top', 30)}",
                json={"query": wiql},
            )
            r.raise_for_status()
            ids = [str(w["id"]) for w in r.json().get("workItems", [])][:200]
            if not ids:
                return "No active work items."
            r = await c.get(
                f"{base}/_apis/wit/workitems",
                params={
                    "ids": ",".join(ids),
                    "api-version": "7.1",
                    "fields": "System.Id,System.Title,System.State,System.WorkItemType,System.AssignedTo",
                },
            )
            r.raise_for_status()
        lines = []
        for w in r.json().get("value", []):
            f = w["fields"]
            who = (f.get("System.AssignedTo") or {}).get("displayName", "unassigned")
            lines.append(f"#{w['id']} [{f['System.WorkItemType']}] {f['System.Title']} - {f['System.State']} ({who})")
        return "\n".join(lines)


class AzdoCreateWorkItem(_AzDO):
    spec = ToolSpec(
        "azdo_create_work_item",
        "Create a work item (Task, Bug, User Story, Issue) in Azure DevOps.",
        _obj(
            {
                **_PROJECT,
                "work_item_type": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            ["project", "work_item_type", "title"],
        ),
        requires_approval=True,
        untrusted_output=False,
    )

    def summarize(self, args: dict[str, Any]) -> str:
        return f'Create {args.get("work_item_type")} in {args.get("project")}: "{args.get("title")}"'

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        base, headers = await self._client_args(ctx)
        patch = [{"op": "add", "path": "/fields/System.Title", "value": args["title"]}]
        if args.get("description"):
            patch.append({"op": "add", "path": "/fields/System.Description", "value": args["description"]})
        url = f"{base}/{quote(args['project'])}/_apis/wit/workitems/${quote(args['work_item_type'])}?api-version=7.1"
        async with ctx.http(headers={**headers, "Content-Type": "application/json-patch+json"}) as c:
            r = await c.post(url, json=patch)
            r.raise_for_status()
            data = r.json()
        return f"Created work item #{data['id']}: {data.get('_links', {}).get('html', {}).get('href', '')}"


class AzdoListPipelines(_AzDO):
    spec = ToolSpec("azdo_list_pipelines", "List pipelines in an Azure DevOps project.", _obj(_PROJECT, ["project"]))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        base, headers = await self._client_args(ctx)
        async with ctx.http(headers=headers) as c:
            r = await c.get(f"{base}/{quote(args['project'])}/_apis/pipelines", params={"api-version": "7.1"})
            r.raise_for_status()
        return (
            "\n".join(f"{p['id']}: {p['name']} ({p.get('folder', '')})" for p in r.json().get("value", []))
            or "No pipelines."
        )


class AzdoListBuilds(_AzDO):
    spec = ToolSpec(
        "azdo_list_builds",
        "List recent builds and their results in an Azure DevOps project.",
        _obj({**_PROJECT, "top": {"type": "integer"}}, ["project"]),
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        base, headers = await self._client_args(ctx)
        async with ctx.http(headers=headers) as c:
            r = await c.get(
                f"{base}/{quote(args['project'])}/_apis/build/builds",
                params={"api-version": "7.1", "$top": args.get("top", 15)},
            )
            r.raise_for_status()
        return (
            "\n".join(
                f"{b['buildNumber']} {b['definition']['name']} {b['status']}/{b.get('result', '-')} "
                f"{b.get('sourceBranch', '')} {b.get('finishTime', '')}"
                for b in r.json().get("value", [])
            )
            or "No builds."
        )


DEV_TOOLS: list[type[Tool]] = [
    GitHubListIssues,
    GitHubCreateIssue,
    GitHubSearchCode,
    GitHubGetFile,
    GitHubListWorkflowRuns,
    AzdoListWorkItems,
    AzdoCreateWorkItem,
    AzdoListPipelines,
    AzdoListBuilds,
]
