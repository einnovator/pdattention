# PRA CLI cross-machine end-to-end evidence

Both runs used commit `9700c402be7476f6429935c49d02d127bc9472ff` and enabled
live Hugging Face catalog search plus immutable bundle pull. The suite executes
every public leaf command in an isolated subprocess and records both its help
contract and an expected semantic outcome.

| Host | Platform | Python | Torch | Accelerator | Duration | Help | Semantics | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `DESKTOP-9IOCN3C` | Windows AMD64 | 3.10.11 | 2.12.1+cu126 | CUDA | 520.10 s | 50/50 | 50/50 | PASS |
| `Mac` | macOS arm64 | 3.10.20 | 2.8.0 | MPS | 141.25 s | 50/50 | 50/50 | PASS |

The machine-readable receipts are `windows_local.json` and
`macos_remote.json`. Each contains 100 bounded command receipts with arguments,
classification, exit status, duration, and output excerpt. No credential or
token value is recorded.

The cross-platform run found and fixed two portability defects before these
receipts were produced:

1. Bundle tree fingerprints depended on Windows versus POSIX path ordering.
2. Gateway startup could block on reverse DNS while binding a loopback address.
