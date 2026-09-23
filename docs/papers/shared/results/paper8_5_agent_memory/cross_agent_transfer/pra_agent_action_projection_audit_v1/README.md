# PRA Agent action-projection audit

This diagnostic isolates a host-protocol defect found while qualifying the
bounded inspection guard on Easy-14 Task 4 (`django__django-15741`).  The first
guarded execution correctly rejected read-only actions after eight executed
tools, but the durable assistant projection had already labelled each rejected
proposal as `[PRA action executed]`.  The next user record labelled the same
proposal rejected.  Qwen3-Coder-30B then proposed four more read-only actions
and produced no patch.

The run was stopped and is not a quality or savings observation.  Commit
`cfb39826` moves the execution projection after guard acceptance and records a
rejected proposal as `[PRA action rejected] ... but was not executed`.  The
host rejection remains a separate role-valid record.  Regression tests require
that rejected actions never carry the executed marker.

The first post-fix retry failed transiently at the remote model endpoint after
seven executed tools, before reaching the guard.  It is also quarantined and
does not test the repair.  A fresh immutable retry follows after three
successful deterministic health probes.

