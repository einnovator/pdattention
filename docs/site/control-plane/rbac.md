# Control Plane RBAC and Audit

| Role | Read fleet and Registry | Engine actions/config | Registry write | Approvals | Fleet/identity administration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Viewer | Yes | No | No | No | No |
| Operator | Yes | Yes | No | No | No |
| Approver | Yes | No | Yes | Yes | No |
| Administrator | Yes | Yes | Yes | Yes | Yes |

The same manager permission checks guard browser actions, REST requests, MCP,
the embedded CLI, and PRA Agent tools. Adapter checks are defense in depth; a
caller cannot bypass authorization by invoking a manager directly. A Viewer
cannot convert a conversational request into an engine mutation.
This release leaves room for later attribute-based restrictions without
pretending that they are already implemented.

Every mutation appends an audit row containing actor, roles, permission,
operation, target, before/after state where available, timestamp, reason,
request ID, trace ID, transport, idempotency key, and result.
Failures are audited too. The UI always asks for a reason, and destructive or
high-impact operations require explicit confirmation.

Audit records are append-only through the application interface. Database
administrators should enforce retention, backup, and write privileges at the
PostgreSQL layer according to local compliance requirements.
