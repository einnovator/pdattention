# Hugging Face Integration

Hub import reads only repository metadata and the small PRA bundle manifest. It
resolves the requested revision to a commit SHA and does not mirror base-model
weights.

```bash
pra registry import-hf EInnovator/pra-qwen3-0.6b
pra registry import-hf EInnovator/pra-qwen3-4b-mlx-4bit --revision 49c1867...
pra registry sync-hf-collection EInnovator/progressive-retrieval-attention
```

Private repositories use the normal `huggingface_hub` credential chain. The
Registry records only a credential reference and whether the source is private;
it never returns or audits the token. Collection synchronization reports each
item independently, so one malformed bundle does not hide successful imports.

Filesystem import uses the same connector contract. S3, OCI, Artifactory,
MLflow, and custom enterprise connectors implement `ArtifactConnector.inspect`
and return normalized model and bundle metadata.
