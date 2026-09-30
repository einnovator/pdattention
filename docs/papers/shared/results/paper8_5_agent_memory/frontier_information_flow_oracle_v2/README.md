# Boundary-free frontier DAG audit v2

This bundle audits the recent-frontier DAG without providing evaluator task
IDs or episode boundaries to the selector. User prompts are observable graph
roots. Resource/version and declared-dependency edges may reconnect records
across any number of prompt intervals.

## Inputs

- Five-issue persistent prefix copied from the locked Qwen3-Coder-30B FULL
  campaign.
- Sixth completed trajectory copied byte-for-byte from the original `.8`
  campaign output (`sha256:059bd873b2527c4179adcc7b9d39d4ef1a27695312617afa4e037f69f547635b`).
- Qwen3-Coder tokenizer revision
  `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`.

## Structural findings

For the six independent trajectories, the evaluator-only independence oracle
and the heuristic no-path rule agree exactly for both two- and three-prompt
frontiers: token-weighted precision and recall are 1.0, with zero false-positive
or false-negative retired groups. There are no cross-interval resource edges.
The M2/P1 edge audit observes one bounded protocol-control edge per active
request and zero unexplained cross-interval edges.

The synthetic controls now distinguish the cases required by the claim:

- disjoint resources: no cross-interval edge and exact oracle agreement;
- same resource in one declared workspace: one resource-flow edge and exact
  oracle agreement;
- same path in distinct declared workspaces: no false dependency;
- unknown effect: candidate confidence is downgraded rather than certified.

## Important invalidity found

Resource reachability alone is not sufficient for a valid model transcript.
With every user prompt pinned, whole-causal-group retirement leaves prior
closed prompts visibly unanswered:

| Frontier | Whole-group saving | Orphaned closed prompts | Observation-payload saving | Action+observation saving |
|---|---:|---:|---:|---:|
| M2 | 39.30% | 3 | 18.49% | 33.78% |
| M3 | 35.47% | 2 | 17.09% | 30.80% |

The payload-only and action-plus-observation realizations preserve conversation
topology and produce zero orphaned closed prompts, but the current ordinary-text
implementation uses replacement text. It therefore cannot be presented as
native-K/V reuse. A Paper 4.5 realization must select original K/V spans and
must not re-encode a receipt after the original K/V already exists.

The protocol-exemplar floor does not repair this defect. In this N=6 prefix,
its selected exemplar belongs to the recent frontier; it does not close the
three older pinned prompts.

## Consequence

Whole-group M2/P1 is structurally invalid as a production profile even though
its resource-retirement precision is perfect on this independent sequence.
The implemented successor retires a complete older prompt component only when
(1) terminal evidence exists and (2) every causal group in that component is
disconnected from the live prompt frontier. The selector receives no task ID;
the component is induced by observable prompt roots and information-flow
reachability. On the frozen N=6 prefix, atomic M2 retires three closed
components, saves 29.45%, and leaves zero orphaned prompts. Atomic M3 retires
two components, saves 21.13%, and likewise leaves zero orphans. These are
structural opportunities, not autonomous accuracy results.

The imported legacy trajectories lack the newer runtime workspace-lineage
receipt, so these independent-component decisions remain in the explicit
`heuristic_allowed` tier. The `certified_only` arm correctly abstains. New
autonomous runs must capture workspace lineage and resource versions before a
certified deployment claim.

## Files

- `evidence_n5.json`: five-issue audit.
- `evidence_n6.json`: six-issue audit without exemplar floors.
- `evidence_n6_m2p1.json`: six-issue M2/P1 audit.
- `cross_interval_audit_n6_m2_p1.json`: request-by-request cross-edge ledger.
- `task6_persistent_episode_export.json`: frozen sixth trajectory input.
