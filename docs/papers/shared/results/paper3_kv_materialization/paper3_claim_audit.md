# Paper 3 Claim Audit

This audit maps each main claim to the frozen artifact that supports it. It separates causal
controlled evidence, paired pretrained estimates, preserved pilot observations, and engineering
boundaries.

| Claim | Evidence | Status / boundary |
|---|---|---|
| Exact evidence is content-causally useful under fixed selection and K/V budget. | `toy_materialization/toy_materialization_rows.csv`: T1 minus T9 margin `+4.905`; both average 5.25 K/V tokens. | Supported across five receptive fields and five seeds on the controlled task. |
| Exact cores dominate broader controlled disclosure. | T1 margin gain `+4.107`, accuracy `.720`, evidence mass `.604`; T7 gain `+.981`, accuracy `.300`, evidence mass `.149`. | Supported on this controlled generator; not asserted universally. |
| Training receptive field does not require broader replay here. | `toy_materialization/toy_materialization_frontier.csv` and `toy_radius_by_window.csv`. Exact exceeds parent at every W. | Supported for W=16/32/64/128/global controlled checkpoints. |
| Consumer layer changes whether memory affects output. | `toy_consumer_layer_profile.csv`: exact-core gains 4.107, 2.369, and .554 at layers 0, 2, and 5. | Supported; Qwen layer bands remain inherited rather than tuned on this result. |
| Context changes native value representations. | `toy_portability.csv`: mean layer-2 change `.215`. | Property supported; causal failure interpretation only partial because disclosure correlation is `.026`. |
| Exact Qwen evidence preserves whole-parent likelihood at lower K/V. | `pretrained_confirmation/pretrained_confirmation_paired.csv`: MuSiQue `+.058 [-.446,.518]`, 2Wiki `+.005 [-.245,.202]`; K/V reductions 20.9% and 44.4%. | Supported as no detected held-out degradation, not proof of exact equality or improvement. |
| Smaller Qwen disclosure concentrates evidence attention. | `pretrained_materialization_frontier.csv`: MuSiQue `.556 -> .689`; 2Wiki `.171 -> .208`. | Supported mechanistically on 32 paired identities per dataset. |
| MuSiQue has a dataset-specific memory-consumption mismatch. | Exact memory is `-2.461` nats/token vs no memory; exact preserves parent. Controlled memory and 2Wiki Qwen memory remain useful. | Supported; does not imply a universal frozen-consumer limit. |
| Gist means do not replace token detail. | Controlled T10/T11 and pretrained M0 conditions. | Supported for tested mean-native-K/V gists only. |
| High density does not imply sufficient coverage. | Preserved pilot budget-128 MuSiQue density `.975`, coverage `.533`, likelihood loss. | Supported by the preserved pilot; not pooled with the 32-example confirmation. |
| Materialization improves serving speed. | TTFT is non-monotonic and Python gather remains visible. | Not supported; the paper makes a memory-quality claim only. |

## Protocol Audit

- Controlled selection is one oracle parent in every compared condition.
- T1 and T9 match native K/V tokens and consumer layer.
- Controlled validation and held-out identities are deterministic and disjoint.
- Qwen validation uses 16 identities per dataset; held-out uses 32 per dataset.
- Qwen radius selection is frozen on validation before held-out aggregation.
- Held-out extension IDs exclude every validation-manifest ID.
- All model weights remain frozen; no routing, adapter, LoRA, or serving mechanism is introduced.

## Artifact Boundary

The large controlled JSONL resume log is intentionally excluded from the public artifact set.
Normalized CSV/JSON rows contain every quantity used in the paper. Historical whole-parent
transport byte/timing fields were not exported under the Paper-3 names; the fallback is fixed in
code, and the paper does not reconstruct or compare those missing values.
