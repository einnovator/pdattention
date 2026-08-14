# Paper 2 readability audit

This audit records the final main-text/appendix decision. Scientific artifacts and
receipts remain unchanged.

| Manuscript material | Decision | Reason |
|---|---|---|
| Introduction, architecture diagram, dataset roles | MAIN | Defines the problem, mechanism, and scope. |
| PRA-HF summary and minimal PRA recap | MAIN | Makes the paper readable without Papers 1/1.5. |
| Shared core, thin adapter, and exact retrofit contract | MAIN | Central engineering contribution. |
| Evaluation design and disabled-path parity | MAIN | Required to interpret every claim. |
| Semantic/native representation split | MAIN | Central scientific design rule. |
| Learned-router recall/sparsity frontier | MAIN | Primary discovery result. |
| Eight-stage evaluation hierarchy | MAIN | Separates availability, discovery, representation, materialization, integration, decision, decoding, and behavioral utility. |
| Compact depth and oracle-gap result | MAIN | Strongest causal and mechanistic evidence. |
| QASPER likelihood, decision, completion, and judge headlines | MAIN | Strongest positive pretrained result. |
| Family portability and SDK status headlines | MAIN | Practical contribution and tested boundary. |
| Limitations and Paper 2.5/3/4 handoffs | MAIN | Prevents scope inflation and separates interventions. |
| Full depth/placement tables and route-once traces | APPENDIX | Reproducibility detail after the headline comparison. |
| Full oracle span, layer, K/V, attention, and encoding audit | APPENDIX | Audit receipt supporting the compact decomposition. |
| Residual, LoRA, LM-head, and decision-loss ladders | APPENDIX | Important controls, but chronological and table-heavy. |
| Full behavioral-judge protocol, hashes, reversals, and calibration | APPENDIX | Preserves blinded-evaluation reproducibility. |
| Cross-family and systems benchmark tables | APPENDIX | Supports the portability headline without interrupting it. |
| Routing geometry and failed ablations | APPENDIX | Secondary negative results owned partly by Paper 1.5. |
| Code-level adapter reimplementation guide | APPENDIX | Detailed independent-reproduction path. |
| Repeated routing-versus-payload explanations | REMOVE/MERGE | Consolidated in the architecture and eight-stage hierarchy. |
| Repeated Hotpot one-shot caveats | REMOVE/MERGE | Consolidated as the Paper 2.5 discovery handoff. |
| Repeated likelihood-versus-decoding caveats | REMOVE/MERGE | Consolidated in the QASPER headline and evaluation hierarchy. |
| Repeated SDK-default statements | REMOVE/MERGE | One main recommendation, with detailed receipts in the appendix. |

The restructuring preserves the full evidence trail while ending the main scientific
narrative before the detailed oracle, adaptation, behavioral, portability, and code
appendices.
