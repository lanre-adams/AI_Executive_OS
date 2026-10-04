## How to respond

Reply with exactly ONE JSON object and nothing else.

To use a tool:
{"action": "tool", "tool": "<tool name>", "args": { ... }, "reason": "<one line: why>"}

When you are finished:
{"action": "final", "content": "<your complete deliverable in Markdown>"}

Use tools only when they add information you do not already have. Call one tool per message.
After each tool call you will receive its result and can continue. Finish within the step budget.
