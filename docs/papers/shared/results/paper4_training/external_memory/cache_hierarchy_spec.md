# Cache hierarchy specification

Cold descriptor cache stores versioned metadata and external gists. Source cache stores fetched bytes. Warm native cache stores model/tokenizer/config-fingerprinted K/V and PRA gists. Hot cache stores accelerator selections. Every reuse reauthorizes the URI; source-version changes invalidate source, native, and hot entries.
