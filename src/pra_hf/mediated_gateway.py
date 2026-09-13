"""Location-independent gateway facade backed by the shared request mediator."""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

from .agent_history import AgentRecordizer
from .deployment import PRAEngineAdapter, PRAEngineResult, PRAGatewayMode, PRAWireRequest
from .gateway import FallbackInjectionPolicy, PRAGateway
from .gateway_session import GatewaySessionRegistry
from .mediation import AgentMemoryPlanBuilder, PRAMediationConfig, RequestMediator
from .observability import Observability
from .session_service import SessionService


class MediatedPRAGateway:
    """Apply one logical plan, then delegate to the resolved G00--G11 mode.

    The same facade can run as a standalone proxy or inside an inference
    runtime.  Its policy/location stamp prevents a second active mediator from
    silently changing the request at another hop.
    """

    def __init__(
        self,
        adapter: PRAEngineAdapter,
        *,
        mediation: PRAMediationConfig | Mapping[str, Any],
        recordizer: AgentRecordizer | None = None,
        plan_builder: AgentMemoryPlanBuilder | None = None,
        session_service: SessionService | None = None,
        session_registry: GatewaySessionRegistry | None = None,
        fallback_injection: FallbackInjectionPolicy | str = FallbackInjectionPolicy.BEFORE_CURRENT_USER,
        observability: Observability | None = None,
        bundle_source: str | None = None,
        default_profile: str = "default",
        models: Sequence[str] = (),
    ) -> None:
        self.adapter = adapter
        self.mediator = RequestMediator(
            mediation, recordizer=recordizer, plan_builder=plan_builder
        )
        self.mediation = self.mediator.config
        self.sessions = session_registry or GatewaySessionRegistry(session_service)
        self.models = tuple(models)
        self._default_profile = default_profile
        self._gateways = {
            mode: PRAGateway(
                adapter,
                mode=mode,
                session_registry=self.sessions,
                fallback_injection=fallback_injection,
                observability=observability,
                bundle_source=bundle_source,
                default_profile=default_profile,
                models=models,
            )
            for mode in PRAGatewayMode
        }

    @property
    def default_profile(self) -> str:
        return self._default_profile

    @default_profile.setter
    def default_profile(self, value: str) -> None:
        self._default_profile = str(value)
        for gateway in getattr(self, "_gateways", {}).values():
            gateway.default_profile = self._default_profile

    def capabilities(self) -> dict[str, Any]:
        # G11 exposes the upper bound; request traces disclose the per-request
        # effective mode and any conservative G00/G10 fallback.
        value = self._gateways[PRAGatewayMode.G11_MEDIATION].capabilities()
        value["endpoint_type"] = "mediated-gateway"
        value["gateway_mode"] = "auto" if self.mediation.mode == "auto" else self.mediation.mode
        value["mediation"] = self.mediation.to_dict()
        value["gateway"]["mode"] = value["gateway_mode"]
        return value

    def generate(
        self,
        request: PRAWireRequest | Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
    ) -> PRAEngineResult:
        if not isinstance(request, PRAWireRequest):
            request = PRAWireRequest.from_dict(request)
        prepared = self.mediator.prepare(request, self.adapter.capabilities())
        result = self._gateways[prepared.resolved_mode].generate(
            prepared.request, trace_headers=trace_headers
        )
        return PRAEngineResult(
            result.text,
            result.raw,
            ({"stage": "request_mediation", **dict(prepared.trace)}, *result.trace),
        )

    def stream(
        self,
        request: PRAWireRequest | Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        if not isinstance(request, PRAWireRequest):
            request = PRAWireRequest.from_dict(request)
        prepared = self.mediator.prepare(request, self.adapter.capabilities())

        def rows() -> Iterator[Mapping[str, Any]]:
            yield {
                "type": "trace",
                "request_id": prepared.request.request_id,
                "trace": {"stage": "request_mediation", **dict(prepared.trace)},
            }
            yield from self._gateways[prepared.resolved_mode].stream(
                prepared.request, trace_headers=trace_headers
            )

        return rows()

    def inspect_session(self, tenant_id: str, session_id: str, model: str):
        return self._gateways[PRAGatewayMode.G00_PASS_THROUGH].inspect_session(
            tenant_id, session_id, model
        )

    def close_session(self, tenant_id: str, session_id: str, model: str) -> bool:
        return self._gateways[PRAGatewayMode.G00_PASS_THROUGH].close_session(
            tenant_id, session_id, model
        )


__all__ = ["MediatedPRAGateway"]
