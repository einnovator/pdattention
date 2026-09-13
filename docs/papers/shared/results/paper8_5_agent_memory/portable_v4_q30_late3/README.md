# Portable v4 late-trajectory probe

This bounded frozen-replay probe uses decisions 22--24 from the successful
`django__django-15277` mini-swe-agent trajectory with `qwen3-coder:30b`, the
pinned Qwen tokenizer, temperature zero, top-p one, and seed zero.

The two FULL runs agree on 2/3 exact next commands and first differ only at
decision 24. The two semantic-receipt runs have byte-identical generated
content at all three decisions. Relative to contemporaneous `full_late3.json`,
the canonical post-parser-fix receipt run agrees on 1/3 exact commands and
first differs at decision 22. It materializes 20,893 of 21,907 content tokens:
1,014 tokens saved, or 4.63%. At each decision, H3 replaces three older reads
whose same-version content is covered by later retained reads.

This is an action-agreement probe, not an autonomous task-success result. The
model-visible phrase describing a read as superseded is itself an intervention
and may prime a consolidation action. The predeclared next arm therefore keeps
all H3 evidence out of prompt text and emits only a role-valid empty-result
stub.

`v4_receipt_late3.json` was produced before the frozen-replay fence parser was
fixed. Its decision-22 `generated_command` field is noncanonical: the stored
`generated_content` contains a valid `mswea_bash_command` block. The B file was
created after the fix and is canonical. The generated contents are identical,
so the parser defect does not affect the model output or token accounting.
