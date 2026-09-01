# Publishing PRA Bundles to Hugging Face

Authenticate, build, validate, generate the card, and dry-run before upload:

```bash
pra hf login
pra bundle build .pra/runs/model-calibration -o .pra/bundles/model
pra bundle validate .pra/bundles/model
pra bundle card .pra/bundles/model --update
pra hf push .pra/bundles/model OWNER/pra-model --dry-run
pra hf push .pra/bundles/model OWNER/pra-model \
  --collection OWNER/pra-bundles --tag v0.2.0rc1 --yes
```

Publishing validates the complete bundle and model card before upload. If an
existing repository targets a different base-model identity or immutable base
revision, the command refuses to overwrite it.

## Publish many bundles

```yaml
bundles:
  - bundle: artifacts/pra_hf/bundles/pra-qwen3-0.6b
    repo_id: OWNER/pra-qwen3-0.6b
    collection: OWNER/pra-bundles
    tag: v0.2.0rc1
```

```bash
pra hf publish-manifest releases/pra_bundles.yaml --dry-run
pra hf publish-manifest releases/pra_bundles.yaml --yes
```

Validation is repeatable and each bundle receives an independent result. Normal
pull-request CI performs only validation and dry runs. Publishing requires a
maintainer-controlled manual release with protected credentials.

## Verify the release

```bash
pra hf pull OWNER/pra-model
pra bundle inspect OWNER/pra-model
pra bundle validate OWNER/pra-model
pra bundle resolve BASE_MODEL -e hf -a OWNER/pra-model
```

The pull output includes the immutable Hub revision and cache path. Run a local
`pra evaluate` before promoting the bundle into the trusted registry.
