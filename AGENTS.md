# AGENTS.md

## Distill policy

Use `dx` as the default shell wrapper for noisy non-interactive commands.

Before using `dx`, verify it is available with a real command, not `dx --version`.

Use one of:
- `command -v dx && dx raw true`
- `command -v dx && dx rg rg -n "AGENTS.md" AGENTS.md`

If `dx` is unavailable, do not assume shell functions are loaded. Use the executable on PATH or fall back to explicit raw commands.

- Use `dx rg` for grep/ripgrep commands.
- Use `dx test` for targeted test commands.
- Use `dx lint` for lint/typecheck commands.
- Use `dx check` for repo-wide checks.
- Use `dx diff` for git diff summaries.
- Use `dx raw` or plain commands when exact raw output is required.

For grep/ripgrep, do not send already-correct `path:line:text` output through `distill`.
If there are no matches, return exactly: `NONE`.
- If a command is small and already naturally readable, skip distill.
- If raw output is explicitly requested, skip distill.
- If exact formatting or verbatim logs matter, skip distill.
- If the command is interactive or TUI-based, skip distill.
- Wait for distill to finish before continuing.

For very noisy commands, prefer capture-then-distill behavior instead of streaming live output directly.

In case there is no `dx` mode for what you need, use:
`command 2>&1 | distill "..."`

Rules for distill usage:
- Use strict output contracts, not open-ended prompts.
- Prefer exact schemas, exact line formats, or hard line caps.
- Do not ask distill vague things like "analyze", "identify", or "summarize" unless the output format is fully specified.
- Prefer outputs that are easy to inspect quickly and easy to reuse in follow-up commands.
