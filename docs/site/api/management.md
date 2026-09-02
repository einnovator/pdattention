# Management API Python Reference

The Python API builds the same `pra-management/1` contract used by the CLI and
engine sidecars. Creating configuration or provider objects does not start a
listener; callers must explicitly invoke `start_management_api` or
`serve_management_api` with `enabled=true`.

::: pra_hf.management
