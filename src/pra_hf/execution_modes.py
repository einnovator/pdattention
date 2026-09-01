"""Qualification-driven public execution-mode resolution."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class ExecutionMode(str, Enum):
    """Product terms for how selected information reaches the model."""

    AUTO = "auto"
    SELECTED_CONTEXT = "selected-context"
    NATIVE_MEMORY = "native-memory"
    NATIVE_SERVING = "native-serving"


class ModeStatus(str, Enum):
    """Independent mechanism, quality, economics, and recommendation states."""

    AVAILABLE = "AVAILABLE"
    VALIDATED = "VALIDATED"
    RECOMMENDED = "RECOMMENDED"
    CANDIDATE = "CANDIDATE"
    QUALIFICATION_PENDING = "QUALIFICATION_PENDING"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    BLOCKED = "BLOCKED"
    NOT_MEASURED = "NOT_MEASURED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


_STATUS_MAP = {
    "available": ModeStatus.AVAILABLE,
    "validated": ModeStatus.VALIDATED,
    "measured": ModeStatus.VALIDATED,
    "recommended": ModeStatus.RECOMMENDED,
    "candidate": ModeStatus.CANDIDATE,
    "qualification pending": ModeStatus.QUALIFICATION_PENDING,
    "calibration_pending": ModeStatus.QUALIFICATION_PENDING,
    "research only": ModeStatus.RESEARCH_ONLY,
    "research-only": ModeStatus.RESEARCH_ONLY,
    "blocked": ModeStatus.BLOCKED,
    "not measured": ModeStatus.NOT_MEASURED,
    "not qualified": ModeStatus.QUALIFICATION_PENDING,
    "not applicable": ModeStatus.NOT_APPLICABLE,
    "unavailable": ModeStatus.NOT_APPLICABLE,
}


def normalize_status(value: object) -> ModeStatus:
    return _STATUS_MAP.get(str(value).strip().lower(), ModeStatus.CANDIDATE)


@dataclass(frozen=True)
class ModeEvidence:
    """Four evidence axes for one product mode."""

    mode: ExecutionMode
    mechanism_status: ModeStatus
    quality_status: ModeStatus
    economic_status: ModeStatus
    recommendation_status: ModeStatus
    reason: str

    @property
    def qualifies_for_auto(self) -> bool:
        accepted = {ModeStatus.VALIDATED, ModeStatus.RECOMMENDED}
        return (
            self.mechanism_status in accepted
            and self.quality_status in accepted
            and self.economic_status in accepted
            and self.recommendation_status == ModeStatus.RECOMMENDED
        )

    def to_dict(self) -> dict[str, str]:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, Enum):
                value[key] = item.value
        return value


@dataclass(frozen=True)
class ModeResolution:
    """Auditable result of resolving an explicit or automatic mode request."""

    requested_mode: ExecutionMode
    resolved_mode: ExecutionMode
    candidates: tuple[ModeEvidence, ...]
    reason: str
    fallback: ExecutionMode = ExecutionMode.SELECTED_CONTEXT
    override_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode.value,
            "candidate_modes": [candidate.to_dict() for candidate in self.candidates],
            "resolved_mode": self.resolved_mode.value,
            "reason": self.reason,
            "fallback": self.fallback.value,
            "override_used": self.override_used,
        }

    def explain(self) -> str:
        lines = [f"Requested: {self.requested_mode.value}", ""]
        for candidate in self.candidates:
            lines.extend(
                [
                    candidate.mode.value,
                    f"  mechanism: {candidate.mechanism_status.value.lower()}",
                    f"  quality: {candidate.quality_status.value.lower()}",
                    f"  economics: {candidate.economic_status.value.lower()}",
                    f"  recommendation: {candidate.recommendation_status.value.lower()}",
                    "",
                ]
            )
        lines.extend(
            [
                f"Resolved: {self.resolved_mode.value}",
                f"Reason: {self.reason}",
                f"Fallback: {self.fallback.value}",
            ]
        )
        return "\n".join(lines)


class ExecutionModeResolver:
    """Resolve mode from evidence without treating implementation as qualification."""

    def candidates(self, engine: Mapping[str, Any]) -> tuple[ModeEvidence, ...]:
        capabilities = dict(engine.get("capabilities", {}))
        selected_mechanism = normalize_status(capabilities.get("selected_context", "available"))
        selected = ModeEvidence(
            ExecutionMode.SELECTED_CONTEXT,
            selected_mechanism,
            ModeStatus.VALIDATED if selected_mechanism in {
                ModeStatus.AVAILABLE, ModeStatus.VALIDATED, ModeStatus.RECOMMENDED
            } else selected_mechanism,
            ModeStatus.VALIDATED,
            ModeStatus.RECOMMENDED,
            "Portable selected-text baseline.",
        )
        native = self._native_evidence(engine, ExecutionMode.NATIVE_MEMORY, "native_memory")
        serving = self._native_evidence(engine, ExecutionMode.NATIVE_SERVING, "native_serving")
        return selected, native, serving

    def resolve(
        self,
        requested_mode: ExecutionMode | str,
        engine: Mapping[str, Any],
        *,
        allow_unqualified_native: bool = False,
    ) -> ModeResolution:
        candidates = self.candidates(engine)
        return self.resolve_candidates(
            requested_mode,
            candidates,
            allow_unqualified_native=allow_unqualified_native,
        )

    def resolve_candidates(
        self,
        requested_mode: ExecutionMode | str,
        candidates: Sequence[ModeEvidence],
        *,
        allow_unqualified_native: bool = False,
    ) -> ModeResolution:
        """Resolve precomputed evidence from either a registry or measured run."""

        requested = ExecutionMode(requested_mode)
        candidates = tuple(candidates)
        by_mode = {candidate.mode: candidate for candidate in candidates}
        required = {
            ExecutionMode.SELECTED_CONTEXT,
            ExecutionMode.NATIVE_MEMORY,
            ExecutionMode.NATIVE_SERVING,
        }
        if set(by_mode) != required:
            raise ValueError("Mode evidence must cover all public non-auto modes.")
        if requested == ExecutionMode.AUTO:
            for mode in (ExecutionMode.NATIVE_SERVING, ExecutionMode.NATIVE_MEMORY):
                if by_mode[mode].qualifies_for_auto:
                    return ModeResolution(
                        requested,
                        mode,
                        candidates,
                        f"{mode.value} passed mechanism, quality, incremental economics, and recommendation gates.",
                    )
            native = by_mode[ExecutionMode.NATIVE_MEMORY]
            reason = (
                "Selected Context is the qualified fallback; Native Memory "
                f"economics are {native.economic_status.value.lower()} and Native Serving "
                f"is {by_mode[ExecutionMode.NATIVE_SERVING].mechanism_status.value.lower()}."
            )
            return ModeResolution(requested, ExecutionMode.SELECTED_CONTEXT, candidates, reason)
        candidate = by_mode[requested]
        if requested == ExecutionMode.SELECTED_CONTEXT:
            return ModeResolution(
                requested, requested, candidates, "Selected Context was explicitly requested."
            )
        if candidate.qualifies_for_auto:
            return ModeResolution(
                requested, requested, candidates, f"Qualified {requested.value} was explicitly requested."
            )
        if allow_unqualified_native and candidate.mechanism_status not in {
            ModeStatus.BLOCKED, ModeStatus.NOT_APPLICABLE, ModeStatus.NOT_MEASURED
        }:
            return ModeResolution(
                requested,
                requested,
                candidates,
                f"Unqualified {requested.value} was explicitly allowed for research.",
                override_used=True,
            )
        raise ValueError(
            f"{requested.value} is not qualified for this engine/model environment: "
            f"mechanism={candidate.mechanism_status.value}, "
            f"quality={candidate.quality_status.value}, "
            f"economics={candidate.economic_status.value}, "
            f"recommendation={candidate.recommendation_status.value}."
        )

    def _native_evidence(
        self, engine: Mapping[str, Any], mode: ExecutionMode, capability: str
    ) -> ModeEvidence:
        mechanism = normalize_status(dict(engine.get("capabilities", {})).get(capability, "not measured"))
        quality = (
            ModeStatus.VALIDATED
            if mechanism in {ModeStatus.VALIDATED, ModeStatus.RECOMMENDED}
            else ModeStatus.QUALIFICATION_PENDING
            if mechanism in {ModeStatus.AVAILABLE, ModeStatus.CANDIDATE, ModeStatus.RESEARCH_ONLY}
            else mechanism
        )
        economics = self._economic_status(engine, mode)
        recommendation_text = " ".join(
            str(engine.get(key, ""))
            for key in ("recommended_today", "production_recommendation")
        ).lower()
        mode_phrase = "native serving" if mode == ExecutionMode.NATIVE_SERVING else "native memory"
        recommended = (
            ModeStatus.RECOMMENDED
            if mode_phrase in recommendation_text
            and not any(
                phrase in recommendation_text
                for phrase in (
                    "selected context by default",
                    "prefer selected context",
                    "selected context;",
                    "research-only",
                    "wait for",
                )
            )
            else ModeStatus.CANDIDATE
        )
        return ModeEvidence(
            mode,
            mechanism,
            quality,
            economics,
            recommended,
            "Native modes require measured incremental economics beyond mechanism availability.",
        )

    @staticmethod
    def _economic_status(engine: Mapping[str, Any], mode: ExecutionMode) -> ModeStatus:
        relevant = []
        for metric in engine.get("metrics", ()):
            name = str(metric.get("name", "")).lower()
            if mode == ExecutionMode.NATIVE_SERVING:
                matches = "serving" in name or "scheduler" in name
            else:
                matches = "native/selected" in name or "native economics" in name or "warm native" in name
            if matches:
                relevant.append(metric)
        measured = [
            metric for metric in relevant
            if normalize_status(metric.get("status", "not measured"))
            not in {ModeStatus.NOT_MEASURED, ModeStatus.NOT_APPLICABLE}
            and "not_measured" not in str(metric.get("value", "")).lower()
        ]
        if not measured:
            return ModeStatus.NOT_MEASURED
        ratios = []
        ambiguous = False
        for metric in measured:
            value = str(metric.get("value", ""))
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x", value, re.IGNORECASE)
            if match:
                ratios.append(float(match.group(1)))
            if "interval includes parity" in value.lower() or "approximately" in value.lower():
                ambiguous = True
        if ambiguous or not ratios or (any(value < 1.0 for value in ratios) and any(value > 1.0 for value in ratios)):
            return ModeStatus.QUALIFICATION_PENDING
        return ModeStatus.VALIDATED if all(value < 1.0 for value in ratios) else ModeStatus.BLOCKED
