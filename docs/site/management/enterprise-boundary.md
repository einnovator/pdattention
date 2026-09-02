# Open API and Fleet Control

The PRA Engine Management API is open source and usable independently of
eInnovator Enterprise. It provides local observed state, capability discovery,
and bounded control for one engine instance.

An enterprise fleet service may use the same public contract to aggregate
inventory, distribute desired revisions, detect drift, apply organization
policy, and retain fleet-wide audit history. Those fleet functions do not alter
the local API's license or require applications to send inference traffic
through a commercial service.

This separation is intentional:

| Open local engine API | Enterprise fleet plane |
| --- | --- |
| One engine's observed state | Multi-cluster inventory |
| Local safe actions | Coordinated rollout and policy |
| Local bounded audit | Durable organization audit |
| Local auth and scopes | Identity federation and governance |
| Public OpenAPI contract | Fleet workflows using that contract |
