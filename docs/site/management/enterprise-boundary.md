# Open API and Fleet Control

The PRA Engine Management API is open source and usable independently of
eInnovator Enterprise. It provides local observed state, capability discovery,
and bounded control for one engine instance.

An enterprise fleet service may use the same public contract to aggregate
inventory, distribute desired revisions, detect drift, apply organization
policy, and retain fleet-wide audit history. The separate open
[PRA Registry](../registry/index.md) owns approved metadata and desired state.
An Enterprise Control Plane may add fleet UI, SSO, governance workflows,
policy automation, and cross-cluster audit while both the Registry and Engine
Management APIs remain independently deployable. Those fleet functions do not
alter either open API's license or require applications to send inference
traffic through a commercial service.

This separation is intentional:

| Open local engine API | Enterprise fleet plane |
| --- | --- |
| One engine's observed state | Multi-cluster inventory |
| Local safe actions | Coordinated rollout and policy |
| Local bounded audit | Durable organization audit |
| Local auth and scopes | Identity federation and governance |
| Public OpenAPI contract | Fleet workflows using that contract |
