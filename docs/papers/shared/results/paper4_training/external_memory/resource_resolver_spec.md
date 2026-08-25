# Resource resolver specification

Resolvers implement asynchronous `stat`, `fetch`, and `external_gist` operations. URI schemes are registered as plugins; PRA contains no HTTP-specific branch. `stat` and `fetch` receive an opaque `AuthContext` and must enforce authorization on every access.
