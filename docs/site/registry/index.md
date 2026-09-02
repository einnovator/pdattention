# PRA Registry

The PRA Registry is the open, headless system of record for model identities,
immutable PRA bundles, profiles, compatibility, qualification evidence,
approvals, policies, and optional desired deployment state. It serves engines,
the `pra` CLI, CI, and higher-level control planes through one versioned REST
contract.

The Registry stores **metadata and provenance**, not model weights. Artifact
bytes remain in Hugging Face, private Hub repositories, filesystems, object
stores, OCI registries, Ollama stores, or engine-native stores. Each registry
source records an immutable revision or digest and a credential *reference*;
raw repository credentials are rejected from normal payloads.

## Registry and engine management

These APIs answer different questions:

| Component | Authority | Typical question |
| --- | --- | --- |
| PRA Registry | Desired state and approved metadata | Which immutable bundle/profile should this cluster run? |
| Engine Management API | Observed state of one engine | What is loaded and resident now? |
| Enterprise Control Plane | Fleet workflows and governance | Where is desired state drifting from observed state? |

The Registry is independently usable and open source. A commercial control
plane may add fleet UI, SSO, approval workflow, cross-cluster audit, and policy
automation without changing this public contract.

## Start locally

```bash
pip install -e ".[registry]"
alembic -c alembic.ini upgrade head
pra-registry serve --config registry.yaml
pra registry status
```

SQLite is intended for development. Use PostgreSQL for shared and production
deployments. Swagger is available at `http://127.0.0.1:9200/docs` and ReDoc at
`http://127.0.0.1:9200/redoc`.
