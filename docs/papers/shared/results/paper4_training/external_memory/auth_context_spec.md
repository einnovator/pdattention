# Authorization context specification

Credentials remain provider handles in runtime state. They may flow only from `AuthContext` to a resolver. They must not enter tokenization, prompts, model inputs, logs, checkpoints, reports, or cache keys. Cache identities use tenant/user/session security scope, never raw secrets.
