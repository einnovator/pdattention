# OpenAPI and Swagger

An enabled Python management server publishes:

- Swagger UI at `/docs`
- ReDoc at `/redoc`
- OpenAPI 3 schema at `/openapi.json`

The canonical checked-in schema is [pra-management-v1.json](../api/openapi/pra-management-v1.json).
It is generated from the same FastAPI routes used at runtime:

```bash
python scripts/generate_management_openapi.py
```

CI tests verify the route set, bearer security declaration, response models,
and absence of accidental action parameters. Native/non-Python engine sidecars
must implement this schema rather than introducing an engine-specific API.

Clients should negotiate `pra-management/1` through `/v1/pra/health`, tolerate
additive response fields, and use the URL protocol version rather than guessing
compatibility from the PRA package version.
