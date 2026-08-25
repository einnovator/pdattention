import asyncio
import json

import pytest

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


class MemoryResolver:
    name = "mem"

    def __init__(self):
        self.documents = {}
        self.fetches = 0
        self.credential_calls = 0

    def put(self, uri, text, *, version="v1", public=False):
        self.documents[uri] = {
            "text": text,
            "version": version,
            "public": public,
        }

    def _authorize(self, uri, auth_context):
        document = self.documents[uri]
        credential = auth_context.credentials_for(self.name, uri)
        self.credential_calls += int(credential is not None)
        if not document["public"] and uri not in auth_context.authorization_scopes:
            raise PermissionError(f"Resource not authorized: {uri}")
        return document

    async def stat(self, uri, auth_context, session):
        _ = session
        document = self._authorize(uri, auth_context)
        payload = document["text"].encode()
        return ResourceStat(
            uri=uri,
            resolver=self.name,
            version=document["version"],
            size_bytes=len(payload),
            title=document["text"],
            is_public=document["public"],
        )

    async def fetch(self, uri, auth_context, session, byte_range=None):
        _ = session
        document = self._authorize(uri, auth_context)
        self.fetches += 1
        payload = document["text"].encode()
        if byte_range is not None:
            payload = payload[slice(*byte_range)]
        return payload

    async def external_gist(self, uri, metadata, auth_context, session):
        _ = uri, auth_context, session
        return metadata.title or ""


def encoder(source, metadata, context):
    tokens = source.decode().split()
    return NativeEncoding(
        uri=metadata.uri,
        source_version=metadata.version,
        model_fingerprint=context.model_fingerprint,
        tokenizer_fingerprint=context.tokenizer_fingerprint,
        config_fingerprint=context.config_fingerprint,
        token_count=len(tokens),
        byte_count=len(source) * 4,
        pra_gists=(" ".join(tokens[:3]),),
        logical_offsets=((0, len(tokens)),),
        payload=tuple(tokens),
    )


def materializer(encoding, selected_token_ids):
    tokens = encoding.payload
    if selected_token_ids is not None:
        tokens = tuple(tokens[index] for index in selected_token_ids)
    return HotMemoryHandle(
        uri=encoding.uri,
        source_version=encoding.source_version,
        selected_token_count=len(tokens),
        byte_count=len(tokens) * 4,
        payload=tokens,
    )


def manager_and_resolver():
    resolver = MemoryResolver()
    registry = ResolverRegistry()
    registry.register("mem", resolver)
    manager = ExternalMemoryManager(
        encoding_context=EncodingContext(
            model_fingerprint="model-1",
            tokenizer_fingerprint="tokenizer-1",
            encoding_config={"block_tokens": 32, "position": "rope"},
        ),
        encoder=encoder,
        materializer=materializer,
        resolvers=registry,
    )
    return manager, resolver


def session(uri, *, session_id="s1", user_id="u1", cache_scope="session", secret="token"):
    auth = AuthContext(
        tenant_id="t1",
        user_id=user_id,
        session_id=session_id,
        authorization_scopes=frozenset({uri}),
        credential_provider=lambda resolver, resource: secret,
    )
    return PRASession(
        session_id=session_id,
        user_id=user_id,
        tenant_id="t1",
        auth_context=auth,
        cache_scope=cache_scope,
    )


def test_lazy_admission_moves_cold_to_warm_to_hot_and_reuses_each_cache():
    async def scenario():
        uri = "mem://documents/alpha"
        manager, resolver = manager_and_resolver()
        resolver.put(uri, "alpha evidence connects beta")
        current = session(uri)
        record = await manager.add_reference(current, uri=uri, encoding_mode="lazy")
        assert record.tier == "cold"

        candidates = await manager.route_candidates(current, "alpha evidence")
        admitted = await manager.admit(current, candidates, max_admitted=1)
        assert admitted[0].token_count == 4
        assert record.tier == "warm"
        assert resolver.fetches == 1

        first = await manager.ensure_hot(current, uri)
        second = await manager.ensure_hot(current, uri)
        assert first == second
        assert record.tier == "hot"
        assert manager.metrics.native_hits >= 2
        assert manager.metrics.hot_hits == 1
        assert manager.metrics.snapshot()["native_hit_rate"] > 0

    asyncio.run(scenario())

def test_private_user_cache_cannot_be_reused_by_an_unauthorized_session():
    async def scenario():
        uri = "mem://private/report"
        manager, resolver = manager_and_resolver()
        resolver.put(uri, "private evidence")
        owner = session(uri, user_id="owner", cache_scope="user")
        await manager.add_reference(owner, uri=uri, encoding_mode="eager")

        intruder_auth = AuthContext("t1", "intruder", "s2")
        intruder = PRASession("s2", "intruder", "t1", intruder_auth, cache_scope="user")
        with pytest.raises(PermissionError):
            await manager.add_reference(intruder, uri=uri, encoding_mode="cached_lazy")
        assert uri not in intruder.warm_handles

    asyncio.run(scenario())


def test_global_cache_rejects_private_resources_but_accepts_public_resources():
    async def scenario():
        private_uri = "mem://private/a"
        public_uri = "mem://public/a"
        manager, resolver = manager_and_resolver()
        resolver.put(private_uri, "private", public=False)
        resolver.put(public_uri, "public evidence", public=True)
        with pytest.raises(PermissionError):
            await manager.add_reference(
                session(private_uri),
                uri=private_uri,
                cache_scope="global_public",
            )
        public_auth = AuthContext("t1", "u2", "p1")
        public_session = PRASession("p1", "u2", "t1", public_auth)
        record = await manager.add_reference(
            public_session,
            uri=public_uri,
            encoding_mode="eager",
            cache_scope="global_public",
        )
        assert record.tier == "warm"

    asyncio.run(scenario())


def test_source_version_change_invalidates_source_native_and_hot_state():
    async def scenario():
        uri = "mem://documents/versioned"
        manager, resolver = manager_and_resolver()
        resolver.put(uri, "old evidence", version="v1")
        current = session(uri)
        await manager.add_reference(current, uri=uri, encoding_mode="eager")
        await manager.ensure_hot(current, uri)
        old = current.warm_handles[uri]

        resolver.put(uri, "new replacement evidence", version="v2")
        new = await manager.ensure_warm(current, uri)
        assert old.source_version == "v1"
        assert new.source_version == "v2"
        assert new.payload != old.payload
        assert uri not in current.hot_handles
        assert resolver.fetches == 2

    asyncio.run(scenario())


def test_metadata_only_never_fetches_or_encodes():
    async def scenario():
        uri = "mem://documents/metadata"
        manager, resolver = manager_and_resolver()
        resolver.put(uri, "descriptor only")
        current = session(uri)
        await manager.add_reference(current, uri=uri, encoding_mode="metadata_only")
        with pytest.raises(PermissionError):
            await manager.ensure_warm(current, uri)
        assert resolver.fetches == 0
        assert manager.metrics.newly_encoded_tokens == 0

    asyncio.run(scenario())


def test_credentials_are_opaque_and_absent_from_snapshots_and_metrics():
    async def scenario():
        secret = "do-not-log-this-secret"
        uri = "mem://documents/secret"
        manager, resolver = manager_and_resolver()
        resolver.put(uri, "authorized source")
        current = session(uri, secret=secret)
        await manager.add_reference(current, uri=uri, encoding_mode="eager")
        serialized = json.dumps(
            {
                "session": current.safe_snapshot(),
                "metrics": manager.metrics.snapshot(),
                "auth_repr": repr(current.auth_context),
            }
        )
        assert resolver.credential_calls > 0
        assert secret not in serialized
        assert "credential_provider=<opaque>" in serialized

    asyncio.run(scenario())


def test_session_teardown_removes_only_ephemeral_state():
    async def scenario():
        uri = "mem://documents/ephemeral"
        manager, resolver = manager_and_resolver()
        resolver.put(uri, "ephemeral evidence")
        current = session(uri)
        await manager.add_reference(current, uri=uri, encoding_mode="eager")
        await manager.ensure_hot(current, uri)
        manager.teardown_session(current)
        assert current.closed
        assert current.safe_snapshot()["resource_uris"] == []
        assert not manager.native_cache
        assert not manager.hot_cache
        with pytest.raises(RuntimeError):
            await manager.route_candidates(current, "evidence")

    asyncio.run(scenario())
