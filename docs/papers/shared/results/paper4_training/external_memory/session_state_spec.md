# PRA session state specification

A session isolates registered and admitted resources, warm/hot handles, controller state, budgets, and authorization context. Ephemeral teardown removes session-scoped cache entries and handles; user/tenant/public caches survive only under their declared scopes. Model weights are shared while session memory remains isolated.
