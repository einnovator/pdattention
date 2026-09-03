# OpenCode UX Audit

This audit uses OpenCode's current official [TUI documentation](https://opencode.ai/docs/tui/),
[command reference](https://opencode.ai/docs/commands/), and
[attachment documentation](https://opencode.ai/v2/docs/attachments). PRA adopts
interaction patterns only where they fit typed context and remote fleet use.

| Feature | Earlier PRA | Decision | Status and rationale |
| --- | --- | --- | --- |
| Slash palette | Fixed `if/elif` loop | Adopt | Declarative registry drives dispatch, help, completion, and docs |
| Model switching | Static summary only | Extend | Static and Control Plane targets; native state invalidated on switch |
| Session navigation | Basic list/resume | Retain and extend | Durable session list, new session, JSON/Markdown export |
| Input history | None | Adopt | Bounded persistent history with search and duplicate suppression |
| Nested autocomplete | None | Adopt | Commands and subcommands; dynamic identifiers remain incremental work |
| File attachments | None | Extend | Typed, checksummed, session-scoped resources rather than prompt replay |
| Large paste handling | Inline only | Extend | `user.paste` records preserve full content behind compact rendering |
| Tool activity | Plain output | Adopt incrementally | Compact source/risk views now; collapsible full-screen rendering deferred |
| Cancellation | Exited shell | Adopt | Ctrl+C retains session; Ctrl+D or `/quit` exits |
| Export/share | None | Adopt export | Local Markdown/JSON export; public sharing is intentionally absent |
| Themes/keybindings | Terminal defaults | Partial | `prompt_toolkit` foundation and theme schema; custom map deferred |
| PRA context accounting | Not applicable | PRA-specific | `/context`, `/pra`, typed resources, and selected/native state |
| Distributed fleet | Not applicable | PRA-specific | Qualifications and engine/model targets come from Control Plane REST |

The highest remaining UX work is dynamic identifier completion, a full-screen
model picker, richer streaming event blocks, and configurable keybindings. It
does not block the transport-neutral SDK services or the scriptable CLI.
