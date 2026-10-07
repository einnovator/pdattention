# Same-host logical-policy transfer

Direct MLX and SGLang-MLX consume the same mini-swe-agent Task 1 histories on
the same M4 host. Plain, PRA-100 and matched tail-90 all resolve in six calls
on both engines. Within each arm the assistant-content, command-trajectory and
patch hashes match across engines.

Tail-90 produces the identical logical ledger on both engines: 14,075 selected
of 14,731 full message-content tokens, 14,225 selected K/V tokens and 470
completion tokens. It saves 4.45% on its own materialized trajectory and 4.09%
against paired Full input plus completion. This is a positive low-opportunity
single-task point, not the persistent-session target.

The excluded M5-host control demonstrates why model-visible observations are
part of the pairing identity. A recursive `grep` returned the same lines in a
different order, causing a different edit trajectory and failed grade before
PRA selection was tested. Paper 4.5 owns the corresponding per-engine K/V
movement and temporary-memory accounting.
