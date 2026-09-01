# Bundle Contents

A schema-v2 bundle is a closed, checksummed release:

```text
bundle.yaml
README.md
structural_adapter/
learned_adapters/
profiles/
qualification/
engine_compatibility/
provenance/
```

Small bundles may keep profiles, compatibility, qualification, and provenance
inside `bundle.yaml`. Every referenced local structural or learned component is
copied recursively. Absolute paths, parent-directory escapes, missing payloads,
fingerprint mismatches, and checksum mismatches fail validation.

## Structural adapter

The structural adapter describes layers, attention topology, Q/K/V projection
access, RoPE, grouped or multi-query attention, local/global attention, and
eligible consumer layers. It can be entirely training-free.

## Learned adapters

Routing, query-conditioned, consumer-gate, and memory-use modules are separate
named components. Each records its type, status, training provenance,
validation evidence, base fingerprint, and PRA compatibility. A bundle with no
learned component remains valid.

## Profiles

Profiles select components by stable names. Ordinary users choose a profile;
research-only file overrides remain an advanced workflow.

## Integrity and versioning

`bundle.yaml` records the exact base and tokenizer revision, model fingerprint,
PRA package and schema versions, component fingerprints, and payload checksums.
Changing base compatibility requires a new compatible release or repository;
it must never happen silently under an existing immutable revision.
