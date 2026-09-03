# Interactive TUI

The Agent terminal is built from a declarative `CommandRegistry`, persistent
`HistoryManager`, typed `AttachmentManager`, MCP manager, and inference-target
manager. `prompt_toolkit` provides history navigation and slash completion when
installed; the Click input path remains a compatibility fallback.

```bash
pra agent chat --profile work
```

Use `/` to discover commands, `/mo` to complete model commands, and `/mcp a` to
complete MCP actions. `/history` is entered-input history; `/session export`
exports the logical conversation and event transcript. They are intentionally
separate.

## Typed context

`/attach PATH` stores local-file identity, media type, size, checksum, session,
and text when safely decodable. Binary PDF data is not presented as extracted
text. `/detach ID` appends a tombstone without deleting the source file.

Large input is stored as a `user.paste` record and rendered compactly, while the
full body remains available to PRA selection. Small multiline input stays
inline. `/context` reports logical record types and active target without
dumping resource bodies.

## Operations

`/status` combines session, target, MCP, Control Plane, and attachment state.
`/tools` includes source and risk classification. `/resources` lists identities
without content. `/search` searches the durable transcript and `/last` repeats
the last event. Ctrl+C cancels the current input or turn without deleting the
session; Ctrl+D and `/quit` close the shell.

The current picker is searchable through `/models` and concise model matching.
A full-screen keyboard list is deferred because the same registry and target
manager must also work in noninteractive automation and basic terminals.
