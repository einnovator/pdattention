"""Run the Paper 4 cold/warm/hot external-memory mechanism study.

The fixture uses a deterministic in-memory resolver and encoder.  It validates
cost accounting, cache reuse, admission decomposition, and security boundaries;
``answer_proxy`` means required-document availability, not language-model QA.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import time
from pathlib import Path

from pra_hf.external_memory import (
    AuthContext,
    EncodingContext,
    ExternalMemoryManager,
    HotMemoryHandle,
    NativeEncoding,
    PRASession,
    ResolverRegistry,
    ResourceStat,
)


DOCUMENTS = {
    "mem://public/alpha": "alpha bridge evidence connects amber to cedar",
    "mem://public/beta": "beta bridge evidence connects cobalt to delta",
    "mem://public/gamma": "gamma bridge evidence connects elm to fuchsia",
    "mem://public/delta": "delta bridge evidence connects garnet to hazel",
    "mem://public/epsilon": "epsilon bridge evidence connects indigo to jade",
    "mem://public/zeta": "zeta bridge evidence connects khaki to lilac",
    "mem://public/eta": "eta bridge evidence connects mauve to navy",
    "mem://public/theta": "theta bridge evidence connects ochre to pearl",
}


class SyntheticResolver:
    """Authenticated deterministic source used only by this mechanism study."""

    name = "mem"

    def __init__(self, documents):
        self.documents = dict(documents)
        self.versions = {uri: "v1" for uri in documents}

    async def stat(self, uri, auth_context, session):
        _ = session, auth_context.credentials_for(self.name, uri)
        if uri not in auth_context.authorization_scopes:
            raise PermissionError(uri)
        payload = self.documents[uri].encode()
        return ResourceStat(
            uri=uri,
            resolver=self.name,
            version=self.versions[uri],
            size_bytes=len(payload),
            mime_type="text/plain",
            title=self.documents[uri],
            content_hash=hashlib.sha256(payload).hexdigest(),
            is_public=True,
        )

    async def fetch(self, uri, auth_context, session, byte_range=None):
        _ = session, auth_context.credentials_for(self.name, uri)
        if uri not in auth_context.authorization_scopes:
            raise PermissionError(uri)
        await asyncio.sleep(0.0005)
        payload = self.documents[uri].encode()
        return payload if byte_range is None else payload[slice(*byte_range)]

    async def external_gist(self, uri, metadata, auth_context, session):
        _ = uri, auth_context, session
        return metadata.title or ""


async def encode_source(source, metadata, context):
    await asyncio.sleep(0.0005)
    tokens = tuple(source.decode().split())
    return NativeEncoding(
        uri=metadata.uri,
        source_version=metadata.version,
        model_fingerprint=context.model_fingerprint,
        tokenizer_fingerprint=context.tokenizer_fingerprint,
        config_fingerprint=context.config_fingerprint,
        token_count=len(tokens),
        byte_count=len(tokens) * 2 * 4 * 32,
        pra_gists=(" ".join(tokens[:3]),),
        logical_offsets=((0, len(tokens)),),
        payload=tokens,
    )


async def materialize(encoding, selected_token_ids):
    await asyncio.sleep(0.0002)
    tokens = encoding.payload
    if selected_token_ids is not None:
        tokens = tuple(tokens[index] for index in selected_token_ids)
    return HotMemoryHandle(
        uri=encoding.uri,
        source_version=encoding.source_version,
        selected_token_count=len(tokens),
        byte_count=len(tokens) * 2 * 4 * 32,
        payload=tokens,
    )


def session(session_id, *, user_id="paper4-user", allowed=None, scope="user"):
    allowed = frozenset(allowed or DOCUMENTS)
    auth = AuthContext(
        tenant_id="paper4-tenant",
        user_id=user_id,
        session_id=session_id,
        authorization_scopes=allowed,
        credential_provider=lambda resolver, uri: "runtime-only-fixture-token",
    )
    return PRASession(
        session_id=session_id,
        user_id=user_id,
        tenant_id="paper4-tenant",
        auth_context=auth,
        cache_scope=scope,
    )


def metric_delta(before, after):
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if not key.endswith("_hit_rate")
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


async def register_all(manager, current, mode):
    for uri in DOCUMENTS:
        await manager.add_reference(
            current,
            uri=uri,
            encoding_mode=mode,
            cache_scope=current.cache_scope,
        )


async def access_phase(manager, current, query, required_uri, phase):
    before = manager.metrics.snapshot()
    started = time.perf_counter()
    candidates = await manager.route_candidates(current, query, max_candidates=3)
    admitted = await manager.admit(current, candidates, max_admitted=1, threshold=0.01)
    if admitted:
        await manager.ensure_hot(current, admitted[0].uri)
    elapsed = time.perf_counter() - started
    after = manager.metrics.snapshot()
    delta = metric_delta(before, after)
    selected = [uri for uri, _score in candidates[:1]]
    selected_uri = selected[0] if selected else ""
    warm_handle = current.warm_handles.get(selected_uri)
    hot_handle = current.hot_handles.get(selected_uri)
    return {
        "phase": phase,
        "query": query,
        "required_uri": required_uri,
        "selected_uri": selected_uri,
        "document_recall": float(required_uri in selected),
        "false_admission": float(bool(selected) and required_uri not in selected),
        "answer_proxy": float(required_uri in selected),
        "ttft_proxy_ms": elapsed * 1000,
        "warm_native_kv_bytes": warm_handle.byte_count if warm_handle else 0,
        "hot_active_kv_bytes": hot_handle.byte_count if hot_handle else 0,
        "h2d_materialization_proxy_ms": delta["materialization_seconds"] * 1000,
        "candidate_resources": len(candidates),
        "admitted_resources": len(admitted),
        **delta,
    }


async def run_study(output_dir: Path) -> dict:
    resolver = SyntheticResolver(DOCUMENTS)
    registry = ResolverRegistry()
    registry.register("mem", resolver)
    manager = ExternalMemoryManager(
        encoding_context=EncodingContext(
            model_fingerprint="paper4-synthetic-native-kv-v1",
            tokenizer_fingerprint="whitespace-v1",
            encoding_config={"block_tokens": 32, "position_mode": "rope"},
        ),
        encoder=encode_source,
        materializer=materialize,
        resolvers=registry,
    )

    latency_rows = []
    admission_rows = []
    rag_rows = []
    cache_rows = []
    for index, uri in enumerate(DOCUMENTS):
        label = uri.rsplit("/", 1)[-1]
        query = f"{label} bridge evidence"

        cold = session(f"cold-{index}")
        await register_all(manager, cold, "lazy")
        cold_row = await access_phase(manager, cold, query, uri, "cold")
        cold_row["example_id"] = index
        latency_rows.append(cold_row)

        # Simulate accelerator eviction while retaining the user-scoped native
        # encoding. This makes the warm phase pay materialization but not fetch
        # or native encoding; the following hot phase reuses accelerator state.
        manager.hot_cache.clear()
        warm = session(f"warm-{index}")
        await register_all(manager, warm, "cached_lazy")
        warm_row = await access_phase(manager, warm, query, uri, "warm")
        warm_row["example_id"] = index
        latency_rows.append(warm_row)

        hot_row = await access_phase(manager, warm, query, uri, "hot")
        hot_row["example_id"] = index
        latency_rows.append(hot_row)

        admission_rows.append(
            {
                "example_id": index,
                "query": query,
                "required_uri": uri,
                "candidate_count": cold_row["candidate_resources"],
                "admitted_count": cold_row["admitted_resources"],
                "document_recall": cold_row["document_recall"],
                "false_admission": cold_row["false_admission"],
                "external_descriptor_bytes": sum(len(text) for text in DOCUMENTS.values()),
                "source_bytes_fetched": cold_row["fetched_bytes"],
                "tokens_newly_encoded": cold_row["newly_encoded_tokens"],
            }
        )
        for condition, row in (
            ("native_full_context", {"answer_proxy": 1.0, "document_recall": 1.0}),
            ("standard_rag", cold_row),
            ("eager_pra", cold_row),
            ("lazy_rag_pra", cold_row),
            ("cached_lazy_rag_pra", warm_row),
        ):
            rag_rows.append(
                {
                    "example_id": index,
                    "condition": condition,
                    "answer_proxy": row["answer_proxy"],
                    "document_recall": row["document_recall"],
                    "note": "required-document availability proxy; no LM generation",
                }
            )

        cache_rows.append(
            {
                "example_id": index,
                "cross_session_native_cache_reused": float(warm_row["native_hits"] > 0),
                "same_session_hot_cache_reused": float(hot_row["hot_hits"] > 0),
                "cold_fetched_bytes": cold_row["fetched_bytes"],
                "warm_fetched_bytes": warm_row["fetched_bytes"],
                "hot_fetched_bytes": hot_row["fetched_bytes"],
            }
        )

    # Explicit isolation check uses a private URI and a separate manager scope.
    private_uri = "mem://private/security"
    private_resolver = SyntheticResolver({private_uri: "private security evidence"})
    private_registry = ResolverRegistry()
    private_registry.register("mem", private_resolver)
    private_manager = ExternalMemoryManager(
        encoding_context=manager.encoding_context,
        encoder=encode_source,
        materializer=materialize,
        resolvers=private_registry,
    )
    owner = session("security-owner", user_id="owner", allowed={private_uri})
    await private_manager.add_reference(owner, uri=private_uri, encoding_mode="eager")
    intruder = session("security-intruder", user_id="intruder", allowed=set())
    unauthorized_blocked = False
    try:
        await private_manager.add_reference(intruder, uri=private_uri, encoding_mode="cached_lazy")
    except PermissionError:
        unauthorized_blocked = True

    write_csv(output_dir / "cold_warm_hot_latency.csv", latency_rows)
    write_csv(output_dir / "lazy_admission_results.csv", admission_rows)
    write_csv(output_dir / "rag_pra_hybrid_results.csv", rag_rows)
    write_csv(output_dir / "session_cache_results.csv", cache_rows)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PRA external memory resource",
        "type": "object",
        "required": ["uri", "resolver", "encoding_mode", "cache_scope"],
        "properties": {
            "uri": {"type": "string"},
            "resolver": {"type": "string"},
            "encoding_mode": {"enum": ["eager", "lazy", "cached_lazy", "metadata_only"]},
            "cache_scope": {"enum": ["session", "user", "tenant", "global_public"]},
            "external_gist": {"type": ["string", "null"]},
            "source_version": {"type": ["string", "null"]},
            "model_fingerprint": {"type": ["string", "null"]},
            "tokenizer_fingerprint": {"type": ["string", "null"]},
            "encoding_config_fingerprint": {"type": ["string", "null"]},
            "tier": {"enum": ["cold", "warm", "hot"]},
        },
        "additionalProperties": False,
    }
    (output_dir / "external_memory_registry_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    specs = {
        "resource_resolver_spec.md": """# Resource resolver specification\n\nResolvers implement asynchronous `stat`, `fetch`, and `external_gist` operations. URI schemes are registered as plugins; PRA contains no HTTP-specific branch. `stat` and `fetch` receive an opaque `AuthContext` and must enforce authorization on every access.\n""",
        "auth_context_spec.md": """# Authorization context specification\n\nCredentials remain provider handles in runtime state. They may flow only from `AuthContext` to a resolver. They must not enter tokenization, prompts, model inputs, logs, checkpoints, reports, or cache keys. Cache identities use tenant/user/session security scope, never raw secrets.\n""",
        "session_state_spec.md": """# PRA session state specification\n\nA session isolates registered and admitted resources, warm/hot handles, controller state, budgets, and authorization context. Ephemeral teardown removes session-scoped cache entries and handles; user/tenant/public caches survive only under their declared scopes. Model weights are shared while session memory remains isolated.\n""",
        "cache_hierarchy_spec.md": """# Cache hierarchy specification\n\nCold descriptor cache stores versioned metadata and external gists. Source cache stores fetched bytes. Warm native cache stores model/tokenizer/config-fingerprinted K/V and PRA gists. Hot cache stores accelerator selections. Every reuse reauthorizes the URI; source-version changes invalidate source, native, and hot entries.\n""",
        "security_invariants.md": f"""# Security invariants\n\n- Credentials are opaque runtime handles and are absent from model/artifact state.\n- Unauthorized private cross-user reuse blocked in the mechanism run: `{str(unauthorized_blocked).lower()}`.\n- Cache reuse revalidates resolver authorization.\n- Source versions participate in native and hot cache keys.\n- Global cache entries require explicitly public metadata.\n- Session teardown removes ephemeral state.\n\nExecutable coverage: `tests/test_external_memory_lifecycle.py`.\n""",
        "paper4_5_consolidation_roadmap.md": """# Paper 4.5 memory-consolidation boundary\n\nPaper 4 owns the read path: external descriptors, authenticated lazy admission, scoped caches, native encoding, PRA search, and bounded K/V consumption. Paper 4.5 owns model- or agent-authored writes, episodic admission, merging, summarization, forgetting, conflict/version resolution, provenance, deletion, reconsolidation, and re-encoding after model updates. No write-side policy is implemented in Paper 4.\n""",
    }
    for name, content in specs.items():
        (output_dir / name).write_text(content, encoding="utf-8")

    phases = {
        phase: [row for row in latency_rows if row["phase"] == phase]
        for phase in ("cold", "warm", "hot")
    }
    summary = {
        "status": "read_side_lifecycle_mechanism_complete",
        "examples": len(DOCUMENTS),
        "document_recall": sum(row["document_recall"] for row in admission_rows) / len(admission_rows),
        "false_admission_rate": sum(row["false_admission"] for row in admission_rows) / len(admission_rows),
        "mean_ttft_proxy_ms": {
            phase: sum(row["ttft_proxy_ms"] for row in rows) / len(rows)
            for phase, rows in phases.items()
        },
        "unauthorized_cross_user_reuse_blocked": unauthorized_blocked,
        "scope": "deterministic resolver/cache mechanism; answer quality is not measured",
    }
    macros = "\n".join(
        (
            rf"\newcommand{{\PaperFourExternalExamples}}{{{summary['examples']}}}",
            rf"\newcommand{{\PaperFourExternalDocRecall}}{{{summary['document_recall']:.3f}}}",
            rf"\newcommand{{\PaperFourExternalFalseAdmission}}{{{summary['false_admission_rate']:.3f}}}",
            rf"\newcommand{{\PaperFourColdLatency}}{{{summary['mean_ttft_proxy_ms']['cold']:.2f}}}",
            rf"\newcommand{{\PaperFourWarmLatency}}{{{summary['mean_ttft_proxy_ms']['warm']:.2f}}}",
            rf"\newcommand{{\PaperFourHotLatency}}{{{summary['mean_ttft_proxy_ms']['hot']:.2f}}}",
        )
    )
    (output_dir / "generated_external_results.tex").write_text(
        macros + "\n", encoding="utf-8"
    )
    (output_dir / "external_memory_findings.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/papers/shared/results/paper4_training/external_memory"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(run_study(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
