# PRA frontier report

No PRA frontier is interpreted before a compatible official baseline exists.

Transport qualification, efficacy, equivalence, and product end-to-end rows are reported separately; none is silently promoted into another evidence class.

| Cell | Agent | Evidence role | Selection | Mode | Connection | Engine PRA | Gateway PRA | State | Baseline gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `easy50-gateway-passthrough` | `mini-swe-agent-2.4.6` | `transport_qualification` | `not_applicable` | `gateway_passthrough` | `gateway` | False | False | PENDING | `easy50-no-pra` |
| `easy50-truncation-50` | `mini-swe-agent-2.4.6` | `efficacy` | `not_applicable` | `truncation` | `direct` | False | False | PENDING | `easy50-no-pra` |
| `easy50-pra-selected-50` | `mini-swe-agent-2.4.6` | `efficacy` | `not_applicable` | `gateway_pra` | `gateway` | False | True | PENDING | `easy50-no-pra` |
| `easy50-native-pra-direct-50` | `mini-swe-agent-2.4.6` | `efficacy` | `route_owned` | `native_pra` | `direct` | True | False | PENDING | `easy50-no-pra` |
| `easy50-native-pra-gateway-50` | `mini-swe-agent-2.4.6` | `product_end_to_end` | `route_owned` | `gateway_native_pra` | `gateway` | True | True | BLOCKED | `easy50-no-pra` |
| `easy50-truncation-25` | `mini-swe-agent-2.4.6` | `efficacy` | `not_applicable` | `truncation` | `direct` | False | False | PENDING | `easy50-no-pra` |
| `easy50-pra-selected-25` | `mini-swe-agent-2.4.6` | `efficacy` | `not_applicable` | `gateway_pra` | `gateway` | False | True | PENDING | `easy50-no-pra` |
| `easy50-truncation-12-5` | `mini-swe-agent-2.4.6` | `efficacy` | `not_applicable` | `truncation` | `direct` | False | False | PENDING | `easy50-no-pra` |
| `easy50-pra-selected-12-5` | `mini-swe-agent-2.4.6` | `efficacy` | `not_applicable` | `gateway_pra` | `gateway` | False | True | PENDING | `easy50-no-pra` |
