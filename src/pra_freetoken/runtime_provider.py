"""Remote FreeToken provider with capability-honest E0 defaults."""

from pra_hf.deployment import PRAEngineCapabilities
from pra_hf.engine_profiles import EngineType, PrefixCacheMode
from pra_hf.runtime_providers import RuntimeConfig, RuntimeProvider


class FreeTokenRuntimeProvider(RuntimeProvider):
    engine_type = "freetoken"
    launch_mode = "connect_remote"

    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        del config
        return PRAEngineCapabilities(
            adapter="freetoken_openai",
            engine_type=EngineType.FREETOKEN,
            integration_level="E0",
            prefix_cache_mode=PrefixCacheMode.AUTOMATIC_PREFIX_CACHE,
            automatic_prefix_cache=True,
            session_state=True,
            cache_affinity=True,
            text_fallback=True,
            streaming=False,
        )
