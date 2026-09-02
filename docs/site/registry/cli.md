# Registry CLI

Both entry points operate the same service:

```bash
pra-registry serve --config registry.yaml
pra registry status --registry-url http://127.0.0.1:9200
```

Remote query commands are `models`, `bundles`, `profiles`, `qualifications`,
`deployments`, and `resolve`. All support `--json` or `--yaml`; list operations
also expose bounded `--limit` and `--offset`.

```bash
export PRA_REGISTRY_URL=https://registry.example.com
export PRA_REGISTRY_TOKEN='...'
pra registry bundles --json
pra registry resolve Qwen/Qwen3-14B --engine vllm --trust einnovator-qualified
```

Configuration can also provide `registry.url`. Explicit CLI options override
environment and file values. The local `pra` workflows continue to use their
checked-in bundle catalog when no Registry is configured or reachable; remote
registry use is never an import-time requirement.

See the complete [CLI Command Reference](../cli-reference.md) for every option.
