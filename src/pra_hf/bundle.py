"""Portable, engine-neutral PRA model bundles and Hub distribution."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .bundle_evidence import EvidenceValidationError, validate_bundle_evidence
from .canonical_evidence import (
    CanonicalEvidenceRecord,
    EvidenceCondition,
    MeasurementState,
    MetricGroup,
    render_markdown_table,
)


BUNDLE_SCHEMA_VERSION = 2
MANIFEST_NAMES = ("bundle.yaml", "pra.yaml")
PUBLIC_CARD_SECTIONS = (
    "What this PRA Runtime Bundle is", "Recommended configuration",
    "Headline results", "Evidence by engine, mode, and profile",
    "Installation", "Quickstart", "Profiles",
    "Engine compatibility", "End-to-end qualification",
    "Native Memory qualification", "Research diagnostics",
    "How to evaluate locally", "Known limitations", "Training/creation",
    "Reproducibility", "Community/support",
)


class BundleValidationError(ValueError):
    """Raised when a bundle is incomplete, unsafe, or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: (
            item.relative_to(path).as_posix().casefold(),
            item.relative_to(path).as_posix(),
        ),
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _safe_relative_path(value: object, *, field_name: str) -> Path:
    raw = str(value or "").replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise BundleValidationError(f"{field_name} must be a bundle-relative path: {value!r}")
    if candidate.parts and ":" in candidate.parts[0]:
        raise BundleValidationError(f"{field_name} must not be an absolute Windows path: {value!r}")
    return Path(*candidate.parts)


def _manifest_path(directory: Path) -> Path:
    for name in MANIFEST_NAMES:
        path = directory / name
        if path.is_file():
            return path
    raise BundleValidationError(f"No bundle manifest found in {directory}")


def _snapshot_revision(path: Path) -> str | None:
    parts = path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if len(parts) > index + 1:
            return parts[index + 1]
    return None


@dataclass(frozen=True)
class PRAModelBundle:
    """Resolved structural, learned, profile, evidence, and engine metadata."""

    base_model: Mapping[str, Any]
    structural_adapter: Mapping[str, Any]
    learned_adapters: Mapping[str, Any] = field(default_factory=dict)
    profiles: Mapping[str, Any] = field(default_factory=dict)
    qualification: Mapping[str, Any] = field(default_factory=dict)
    runtime_compatibility: Mapping[str, Any] = field(default_factory=dict)
    engine_realizations: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    checksums: Mapping[str, str] = field(default_factory=dict)
    trust: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = BUNDLE_SCHEMA_VERSION
    local_path: Path | None = field(default=None, compare=False, repr=False)
    source: str | None = field(default=None, compare=False)
    resolved_revision: str | None = field(default=None, compare=False)

    @property
    def benchmark_evidence(self) -> Mapping[str, Any]:
        """Schema-v1 compatibility alias for qualification evidence."""

        return self.qualification

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        revision: str | None = None,
        validate: bool = True,
    ) -> "PRAModelBundle":
        requested = str(source)
        path = Path(source).expanduser()
        resolved_revision = revision
        if not path.exists():
            try:
                from huggingface_hub import HfApi, snapshot_download
            except ImportError as error:
                raise ImportError("Hub bundle resolution requires the 'hf-hub' optional dependency.") from error
            path = Path(snapshot_download(requested, revision=revision))
            resolved_revision = _snapshot_revision(path) or HfApi().model_info(requested, revision=revision).sha
        path = path.resolve()
        value = yaml.safe_load(_manifest_path(path).read_text(encoding="utf-8")) or {}
        bundle = cls(
            base_model=value.get("base_model", value.get("model", {})),
            structural_adapter=value.get("structural_adapter", {}),
            learned_adapters=value.get("learned_adapters", {}),
            profiles=value.get("profiles", {}),
            qualification=value.get("qualification", value.get("benchmark_evidence", {})),
            runtime_compatibility=value.get("runtime_compatibility", {}),
            engine_realizations=value.get("engine_realizations", {}),
            provenance=value.get("provenance", {}),
            checksums=value.get("checksums", {}),
            trust=value.get("trust", {}),
            schema_version=int(value.get("schema_version", 1)),
            local_path=path,
            source=requested,
            resolved_revision=resolved_revision or _snapshot_revision(path),
        )
        if validate:
            bundle.validate()
        return bundle

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_model": dict(self.base_model),
            "structural_adapter": dict(self.structural_adapter),
            "learned_adapters": dict(self.learned_adapters),
            "profiles": dict(self.profiles),
            "qualification": dict(self.qualification),
            "runtime_compatibility": dict(self.runtime_compatibility),
            "engine_realizations": dict(self.engine_realizations),
            "provenance": dict(self.provenance),
            "checksums": dict(self.checksums),
            "trust": dict(self.trust),
        }

    def inspect(self) -> dict[str, Any]:
        value = self.to_dict()
        value.update(
            source=self.source,
            cache_path=str(self.local_path) if self.local_path else None,
            resolved_revision=self.resolved_revision,
            validation_status=self.qualification.get("status", "NOT_MEASURED"),
        )
        return value

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "bundle.yaml"
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    def component_path(self, component: Mapping[str, Any]) -> Path | None:
        """Resolve one packaged component without permitting bundle escape."""

        if not component.get("path"):
            return None
        if self.local_path is None:
            raise BundleValidationError("Component resolution requires a loaded bundle path.")
        relative = _safe_relative_path(component["path"], field_name="component.path")
        resolved = (self.local_path / relative).resolve()
        if self.local_path not in resolved.parents and resolved != self.local_path:
            raise BundleValidationError(f"Component escapes bundle root: {relative}")
        return resolved

    def selected_learned_adapters(self, profile: str | None = None) -> dict[str, Path]:
        """Resolve learned components named by a profile, or all defaults."""

        selected: set[str] = set()
        profile_data = self.profiles.get(profile or "balanced", {})
        if isinstance(profile_data, Mapping):
            for key, value in profile_data.items():
                if key.endswith("_adapter") and value:
                    selected.add(str(value))
        if not selected:
            selected.update(
                name for name, value in self.learned_adapters.items()
                if isinstance(value, Mapping) and bool(value.get("default", False))
            )
        result: dict[str, Path] = {}
        for name in selected:
            component = self.learned_adapters.get(name)
            if isinstance(component, Mapping):
                path = self.component_path(component)
                if path is not None:
                    result[name] = path
        return result

    def validate(self, *, require_card: bool | None = None) -> dict[str, Any]:
        """Validate schema, references, checksums, fingerprints, and model card."""

        errors: list[str] = []
        if self.schema_version not in {1, BUNDLE_SCHEMA_VERSION}:
            errors.append(f"unsupported schema_version={self.schema_version}")
        if not self.base_model.get("id") and self.schema_version >= 2:
            errors.append("base_model.id is required")
        if self.schema_version >= 2:
            revision = str(self.base_model.get("revision", ""))
            if not revision or revision in {"main", "master", "unresolved", "latest"}:
                errors.append("base_model.revision must be immutable")
            if not self.base_model.get("fingerprint"):
                errors.append("base_model.fingerprint is required")
        try:
            validate_bundle_evidence(self)
        except EvidenceValidationError as error:
            errors.append(str(error))
        if self.local_path is not None:
            components: list[tuple[str, Mapping[str, Any]]] = []
            if self.structural_adapter:
                components.append(("structural_adapter", self.structural_adapter))
            components.extend(
                (f"learned_adapters.{name}", value)
                for name, value in self.learned_adapters.items()
                if isinstance(value, Mapping)
            )
            for name, component in components:
                if not component.get("path"):
                    if self.schema_version >= 2:
                        errors.append(f"{name}.path is required")
                    continue
                try:
                    path = self.component_path(component)
                except BundleValidationError as error:
                    errors.append(str(error))
                    continue
                if path is None or not path.exists():
                    errors.append(f"missing referenced component: {name} -> {component.get('path')}")
                elif component.get("fingerprint") and _tree_fingerprint(path) != component["fingerprint"]:
                    errors.append(f"fingerprint mismatch: {name}")
            for relative, expected in self.checksums.items():
                try:
                    path = self.local_path / _safe_relative_path(relative, field_name="checksums key")
                except BundleValidationError as error:
                    errors.append(str(error))
                    continue
                if not path.is_file():
                    errors.append(f"checksummed file is missing: {relative}")
                elif _sha256_file(path) != expected:
                    errors.append(f"checksum mismatch: {relative}")
            card_required = self.schema_version >= 2 if require_card is None else require_card
            card = self.local_path / "README.md"
            if card_required and not card.is_file():
                errors.append("README.md model card is required")
            elif card.is_file() and self.schema_version >= 2:
                try:
                    validate_model_card(card.read_text(encoding="utf-8"), self)
                except BundleValidationError as error:
                    errors.append(str(error))
        if errors:
            raise BundleValidationError("Invalid PRA bundle: " + "; ".join(errors))
        return {
            "status": "VALID", "schema_version": self.schema_version,
            "base_model": dict(self.base_model),
            "components": 1 + len(self.learned_adapters), "checksums": len(self.checksums),
        }


class BundleBuilder:
    """Build a closed, checksummed bundle from onboarding or release metadata."""

    def build(self, run: str | Path, output: str | Path, *, force: bool = False) -> PRAModelBundle:
        source = Path(run).resolve()
        runtime_path = source / "pra.yaml"
        if not runtime_path.is_file():
            runtime_path = source / "bundle.yaml"
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
        target = Path(output).resolve()
        if target.exists() and any(target.iterdir()):
            if not force:
                raise FileExistsError(f"Bundle output is not empty: {target}")
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        manifest_path = source / "manifest.json"
        run_provenance = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        base_model = dict(runtime.get("base_model", runtime.get("model", {})))
        if base_model and not base_model.get("fingerprint"):
            identity = "|".join(str(base_model.get(key, "")) for key in ("id", "revision", "architecture"))
            base_model["fingerprint"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        structural = self._copy_structural(source, target, runtime.get("structural_adapter", {}))
        learned = self._copy_learned(source, target, runtime.get("learned_adapters", {}))
        self._copy_auxiliary(source, target)
        bundle = PRAModelBundle(
            base_model=base_model,
            structural_adapter=structural,
            learned_adapters=learned,
            profiles=runtime.get("profiles", {}),
            qualification=runtime.get("qualification", runtime.get("benchmark_evidence", {})),
            runtime_compatibility=runtime.get("runtime_compatibility", {}),
            engine_realizations=runtime.get("engine_realizations", {}),
            provenance={**run_provenance, **dict(runtime.get("provenance", {}))},
            trust=runtime.get("trust", {"status": "local/private"}),
            schema_version=BUNDLE_SCHEMA_VERSION,
            local_path=target,
            source=str(source),
        )
        bundle = replace(bundle, checksums=self._checksums(target))
        bundle.save_pretrained(target)
        (target / "README.md").write_text(self.model_card(bundle), encoding="utf-8")
        bundle.validate()
        return bundle

    @staticmethod
    def _source_component(source: Path, value: Mapping[str, Any], fallback: Path) -> Path:
        raw = value.get("source_path", value.get("path"))
        if raw:
            relative = _safe_relative_path(raw, field_name="component source path")
            candidate = (source / relative).resolve()
            if source not in candidate.parents and candidate != source:
                raise BundleValidationError(f"Component source escapes run directory: {raw}")
            return candidate
        return fallback

    def _copy_structural(self, source: Path, target: Path, raw: Mapping[str, Any] | None) -> dict[str, Any]:
        value = dict(raw or {})
        component = self._source_component(source, value, source / "structural_adapter")
        if not component.exists():
            raise BundleValidationError(f"Missing structural adapter: {component}")
        destination = target / "structural_adapter"
        self._copy_component(component, destination)
        value.pop("source_path", None)
        value["path"] = "structural_adapter"
        value.setdefault("status", "validated")
        value["fingerprint"] = _tree_fingerprint(destination)
        return value

    def _copy_learned(self, source: Path, target: Path, raw: Mapping[str, Any] | None) -> dict[str, Any]:
        copied: dict[str, Any] = {}
        for name, item in dict(raw or {}).items():
            value = {"path": item} if isinstance(item, str) else dict(item or {})
            component = self._source_component(source, value, source / "learned_adapters" / name)
            if not component.exists():
                raise BundleValidationError(f"Missing learned adapter '{name}': {component}")
            destination = target / "learned_adapters" / str(name)
            self._copy_component(component, destination)
            value.pop("source_path", None)
            value["path"] = f"learned_adapters/{name}"
            value.setdefault("type", name)
            value.setdefault("status", "candidate")
            value["fingerprint"] = _tree_fingerprint(destination)
            copied[str(name)] = value
        return copied

    @staticmethod
    def _copy_component(source: Path, destination: Path) -> None:
        if source.is_symlink():
            raise BundleValidationError(f"Symlinked bundle components are not accepted: {source}")
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=False)
        elif source.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)
        else:
            raise BundleValidationError(f"Bundle component is not a file or directory: {source}")

    def _copy_auxiliary(self, source: Path, target: Path) -> None:
        """Copy optional release evidence and metadata into the closed bundle."""

        for name in ("profiles", "qualification", "engine_compatibility", "provenance"):
            component = source / name
            if component.exists():
                self._copy_component(component, target / name)

    @staticmethod
    def _checksums(target: Path) -> dict[str, str]:
        return {
            path.relative_to(target).as_posix(): _sha256_file(path)
            for path in sorted(target.rglob("*"))
            if path.is_file() and path.name not in MANIFEST_NAMES and path.name != "README.md"
        }

    @staticmethod
    def model_card(bundle: PRAModelBundle) -> str:
        """Render a public, evidence-bounded Hugging Face model card."""

        model = bundle.base_model.get("id", "unknown")
        revision = bundle.base_model.get("revision", "unresolved")
        architecture = bundle.base_model.get("architecture", "NOT_MEASURED")
        parameters = bundle.base_model.get("parameter_count", bundle.base_model.get("parameter_count_approx", "NOT_MEASURED"))
        post_training = bundle.base_model.get("post_training")
        tokenizer_revision = bundle.base_model.get("tokenizer_revision", revision)
        datasets = sorted({str(row.get("dataset")) for row in _qualification_rows(bundle) if row.get("dataset") and row.get("dataset") != "NOT_MEASURED"})
        metadata: dict[str, Any] = {
            "library_name": "pra", "base_model": model,
            "tags": ["pra", "progressive-retrieval-attention", "adapter", "long-context"],
        }
        if datasets:
            metadata["datasets"] = datasets
        if bundle.provenance.get("license"):
            metadata["license"] = bundle.provenance["license"]
        repo = bundle.provenance.get("hf_repo", "OWNER/REPO")
        collection = bundle.provenance.get("hf_collection")
        publisher = str(bundle.trust.get("publisher", "")).strip()
        preferred_engine = (
            "mlx"
            if "mlx" in bundle.runtime_compatibility
            and str(model).lower().startswith("mlx-community/")
            else "hf"
        )
        qualification = bundle.qualification if isinstance(bundle.qualification, Mapping) else {}
        headline = [row for row in qualification.get("headline", []) if isinstance(row, Mapping)]
        recommended_name, recommended = next(
            ((str(name), value) for name, value in bundle.profiles.items()
             if isinstance(value, Mapping) and value.get("recommended") is True),
            ("balanced", bundle.profiles.get("balanced", {})),
        )
        engine_name = str(recommended.get("engine", preferred_engine)) if isinstance(recommended, Mapping) else preferred_engine
        mode = str(recommended.get("mode", "Selected Context")) if isinstance(recommended, Mapping) else "Selected Context"
        native_status = "NOT_MEASURED"
        engine_value = bundle.runtime_compatibility.get(engine_name, {})
        if isinstance(engine_value, Mapping):
            native_status = str(engine_value.get("native_memory", native_status))
        title_suffix = bundle.base_model.get("quantization", {})
        quantization = (
            f"{title_suffix.get('bits')}bit"
            if isinstance(title_suffix, Mapping) and title_suffix.get("bits")
            else str(title_suffix.get("name", ""))
            if isinstance(title_suffix, Mapping)
            else str(title_suffix or "")
        )
        title_engine = " / ".join(value for value in (engine_name.upper(), quantization) if value)
        lines = ["---", yaml.safe_dump(metadata, sort_keys=False).strip(), "---", "", f"# PRA Runtime Bundle for {model} · {title_engine}", ""]
        lines += [
            "## What this PRA Runtime Bundle is", "",
            "This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.", "",
            f"- Base model: `{model}`", f"- Immutable revision: `{revision}`",
            f"- Architecture: `{architecture}`", f"- Parameters: `{parameters}`",
            f"- Tokenizer revision: `{tokenizer_revision}`", "",
            "## Recommended configuration", "",
            f"- Engine: **{engine_name}**", f"- Recommended PRA mode: **{mode}**",
            f"- Recommended profile: **{recommended_name.upper()}**",
            f"- Bundle evidence tier: **{qualification.get('status', 'NOT_MEASURED')}**",
            f"- Native Memory status: **{native_status}**", "",
            "Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.", "",
            "## Headline results", "",
        ]
        if post_training:
            lines.insert(lines.index("## Recommended configuration") - 1, f"- Post-training: `{post_training}`")
        if headline:
            lines += ["| Workload | Baseline quality | PRA quality | Quality Δ | Input/context Δ | TTFT Δ | Completion Δ | Paired parity | Evidence |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
            for row in headline:
                baseline, pra = row.get("baseline", {}), row.get("pra", {})
                delta = row.get("deltas", {})
                parity = row.get("semantic_equivalence", {})
                metric_name = baseline.get("quality_metric", "metric")
                lines.append(
                    f"| {row.get('dataset', row.get('workload', 'combined'))} (n={row.get('sample_count', 'NOT_MEASURED')}) "
                    f"| {metric_name}={_metric(baseline.get('quality'))} | {metric_name}={_metric(pra.get('quality'))} "
                    f"| {_signed(delta.get('quality'))} | {_signed_pct(delta.get('visible_tokens_pct'))} "
                    f"| {_signed_pct(delta.get('ttft_pct'))} | {_signed_pct(delta.get('completion_latency_pct'))} "
                    f"| {parity.get('exact_output_pairs', 'NOT_MEASURED')}/{parity.get('paired_examples', 'NOT_MEASURED')} "
                    f"| {row.get('evidence_tier', 'NOT_MEASURED')} |"
                )
            receipt = headline[0]
            lines += [
                "", "All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.", "",
                "Evidence receipt: "
                f"`{receipt.get('engine')} {receipt.get('engine_version')}`; "
                f"{receipt.get('hardware')}; {receipt.get('cohort')} (n={receipt.get('sample_count')}); "
                f"{receipt.get('date')}; PRA commit `{receipt.get('pra_commit')}`; "
                f"artifact `{receipt.get('artifact')}`; SHA-256 `{receipt.get('artifact_sha256')}`.", "",
            ]
        else:
            lines += ["No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.", ""]
        runtime_smoke = qualification.get("runtime_smoke")
        if isinstance(runtime_smoke, Mapping):
            smoke_memory = runtime_smoke.get("memory", {})
            peak_bytes = (
                smoke_memory.get("peak_memory_bytes")
                or smoke_memory.get("cuda_peak_allocated_bytes")
            ) if isinstance(smoke_memory, Mapping) else None
            runtime = runtime_smoke.get("runtime", {})
            lines += [
                "## Exact-identity runtime smoke", "",
                "This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.", "",
                "| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |",
                "| --- | --- | ---: | ---: | ---: | --- |",
                f"| {runtime_smoke.get('status', 'NOT_MEASURED')} | {runtime.get('hardware', 'NOT_MEASURED')} "
                f"| {_metric(runtime_smoke.get('load_seconds'))} s | {_metric(runtime_smoke.get('generation_seconds'))} s "
                f"| {_bytes(peak_bytes)} | {runtime_smoke.get('claim_scope', 'runtime smoke')} |", "",
                "Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.", "",
            ]
        canonical_records = _canonical_evidence_rows(bundle)
        lines += [
            "## Evidence by engine, mode, and profile", "",
            "Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.", "",
            "| Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Measured metric groups |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, item in bundle.profiles.items():
            value = item if isinstance(item, Mapping) else {}
            profile_engine = str(value.get("engine", engine_name))
            profile_mode = str(value.get("mode", mode))
            matches = [
                record for record in canonical_records
                if record.key.profile.lower() == str(name).lower()
                and record.key.mode.replace("_", "-").lower() == profile_mode.replace(" ", "-").replace("_", "-").lower()
                and (
                    record.key.engine.lower() == profile_engine.lower()
                    or record.key.engine.lower().startswith(profile_engine.lower() + "-")
                )
            ]
            groups = sorted({
                definition.group.value
                for record in matches
                for metric, definition in record.metric_definitions.items()
                if any(
                    evidence.metrics.get(metric) is not None
                    and evidence.metrics[metric].state.value == "MEASURED"
                    for evidence in record.conditions.values()
                )
            })
            profile_status = str(value.get("status", "NEEDS_RUN")).upper()
            routing_adapter = value.get("routing_adapter")
            def fallback(condition: EvidenceCondition) -> str:
                if profile_status == "CALIBRATION_PENDING":
                    return "CALIBRATION_PENDING"
                if condition == EvidenceCondition.PRA_ADAPTOR_BUNDLE:
                    if routing_adapter:
                        return "NEEDS_RUN"
                    return (
                        "NOT_APPLICABLE"
                        if bundle.learned_adapters
                        else "NO_QUALIFIED_ADAPTER"
                    )
                return "NEEDS_RUN"

            lines.append(
                f"| {profile_engine} | {profile_mode} | {str(name).upper()} "
                f"| {_condition_coverage(matches, EvidenceCondition.NO_PRA, fallback(EvidenceCondition.NO_PRA))} "
                f"| {_condition_coverage(matches, EvidenceCondition.PRA_NO_ADAPTOR, fallback(EvidenceCondition.PRA_NO_ADAPTOR))} "
                f"| {_condition_coverage(matches, EvidenceCondition.PRA_ADAPTOR_BUNDLE, fallback(EvidenceCondition.PRA_ADAPTOR_BUNDLE))} "
                f"| {', '.join(groups) if groups else fallback(EvidenceCondition.PRA_NO_ADAPTOR)} |"
            )
        lines += ["", "## Canonical three-condition evidence", ""]
        if canonical_records:
            lines += [
                "Each table holds task, hardware, engine, model, mode, and profile fixed. Deltas are candidate minus No PRA and retain their mathematical sign.", "",
            ]
            for record in canonical_records:
                lines += [
                    f"### {record.key.task} / {record.key.engine} / {record.key.profile}", "",
                    f"Exact identity: `{record.key.model_id}` at `{record.key.model_revision}` on `{record.key.hardware}`.", "",
                ]
                for group in MetricGroup:
                    if not any(metric.group == group for metric in record.metric_definitions.values()):
                        continue
                    lines += [
                        f"#### {group.value.title()}", "",
                        render_markdown_table(record, group, compact_missing=True).rstrip(), "",
                    ]
        else:
            lines += [
                "A complete matched No PRA / PRA - No Adaptor / PRA - Adaptor Bundle cohort is not packaged for this exact identity.", "",
                "| Condition | Evidence status |",
                "| --- | --- |",
                "| No PRA | `NEEDS_RUN` |",
                "| PRA - No Adaptor | `NEEDS_RUN` |",
                f"| PRA - Adaptor Bundle | `{'NEEDS_RUN' if bundle.learned_adapters else 'NO_QUALIFIED_ADAPTER'}` |", "",
                "Existing selector-frozen Selected Context versus Native Memory measurements remain reported below as transport evidence; they are not silently relabeled as adaptor evidence.", "",
            ]
        lines += [
            "## Installation", "", "```bash", "pip install 'pra-hf[hf-hub,hf-runtime]'", "pra doctor", "```", "",
            "## Quickstart", "", "```bash", f"pra inspect {model} -e {preferred_engine} -a {repo}",
            f"pra evaluate {model} -e {preferred_engine} -D qasper -a {repo}", "pra recommend .pra/runs/latest",
            f"pra serve {model} -e {preferred_engine} -a {repo} -p {recommended_name}", "```", "",
            "## Profiles", "", "| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |", "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, item in bundle.profiles.items():
            value = item if isinstance(item, Mapping) else {}
            consumers = value.get("consumer_layers", "all eligible")
            if isinstance(consumers, Sequence) and not isinstance(consumers, str):
                consumers = ", ".join(str(layer) for layer in consumers)
            routing = value.get("routing_adapter") or "generic cosine"
            recommendation = "Default" if value.get("recommended") else value.get("recommendation", "Not promoted")
            lines.append(f"| {str(name).upper()} | {value.get('purpose', 'General PRA use')} | {routing} | {consumers} | {value.get('status', 'NOT_MEASURED')} | {recommendation} |")
        lines += ["", "## Engine compatibility", "", "| Engine | Selected Context | Native Memory | Native Serving | Recommended today |", "| --- | --- | --- | --- | --- |"]
        for engine, item in bundle.runtime_compatibility.items():
            value = item if isinstance(item, Mapping) else {}
            lines.append(f"| {engine} | {value.get('selected_context', 'NOT_MEASURED')} | {value.get('native_memory', 'NOT_MEASURED')} | {value.get('native_serving', 'NOT_MEASURED')} | {value.get('recommended', 'Selected Context')} |")
        lines += ["", "## End-to-end qualification", ""]
        end_task = [row for row in _qualification_rows(bundle) if row.get("metric_class") == "END_TASK"]
        if end_task:
            lines += ["| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |", "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
            for row in end_task:
                for label, values in (("Selected Context", row.get("baseline", {})), ("Native Memory", row.get("pra", {}))):
                    lines.append(f"| {row.get('dataset')} (n={row.get('sample_count')}) | {label} | {values.get('quality_metric', 'metric')}={_metric(values.get('quality'))} | {_metric(values.get('visible_tokens'))} | {_metric(values.get('ttft_ms', {}).get('p50'))} ms | {_metric(values.get('completion_latency_ms', {}).get('mean'))} ms | {row.get('hardware')} | {row.get('evidence_tier')} |")
        else:
            lines.append("What remains to be measured: paired end-task quality for this exact bundle identity.")
        lines += ["", "## Native Memory qualification", ""]
        if headline:
            lines += [
                "Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.", "",
                "| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for row in end_task:
                baseline, pra = row.get("baseline", {}), row.get("pra", {})
                baseline_ms = baseline.get("completion_latency_ms", {}).get("mean")
                pra_ms = pra.get("completion_latency_ms", {}).get("mean")
                ratio = pra_ms / baseline_ms if baseline_ms and pra_ms is not None else None
                lines.append(
                    f"| {row.get('dataset')} | {_metric(pra.get('selected_native_kv_tokens'))} "
                    f"| {_bytes(pra.get('active_detail_bytes'))} | {_bytes(pra.get('peak_memory_bytes'))} "
                    f"| {_metric(ratio)}x |"
                )
        else:
            lines.append("What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.")
        lines += ["", "## Research diagnostics", ""]
        diagnostics = [row for row in _qualification_rows(bundle) if row.get("metric_class") == "ROUTING_DIAGNOSTIC"]
        if diagnostics:
            lines += ["| Dataset | Router/profile | Metric | Value | Cohort | Evidence |", "| --- | --- | --- | ---: | ---: | --- |"]
            for row in diagnostics:
                lines.append(f"| {row.get('dataset')} | {row.get('profile')} | {row.get('quality_metric')} | {_metric(row.get('quality'))} | {row.get('sample_count')} | {row.get('evidence_tier')} |")
        else:
            lines.append("No separate routing diagnostic is packaged for this bundle.")
        lines += [
            "", "These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.", "",
            "## How to evaluate locally", "", "```bash", f"pra evaluate {model} -e {preferred_engine} -a {repo} -D qasper -o .pra/runs/qasper", "pra recommend .pra/runs/qasper", "pra report .pra/runs/qasper --format html", "```", "",
            "## Known limitations", "",
        ]
        limitations = bundle.qualification.get("limitations", []) if isinstance(bundle.qualification, Mapping) else []
        lines.extend(f"- {item}" for item in limitations or ["Unlisted engine, quantization, tokenizer, and hardware combinations are not qualified.", "Smoke evidence does not establish production-scale quality or tail latency."])
        lines += ["", "## Training/creation", ""]
        training = bundle.qualification.get("training", {}) if isinstance(bundle.qualification, Mapping) else {}
        if training:
            lines.extend(f"- {key.replace('_', ' ').title()}: `{value}`" for key, value in training.items())
        else:
            lines.append("The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.")
        lines += [
            "", "## Reproducibility", "", f"- PRA commit: `{bundle.provenance.get('pra_commit', 'NOT_MEASURED')}`",
            f"- Bundle build commit: `{bundle.provenance.get('bundle_build_commit', 'NOT_MEASURED')}`", f"- Bundle schema: `{bundle.schema_version}`",
            f"- PRA package: `{bundle.provenance.get('pra_version', 'NOT_MEASURED')}`", "- Component fingerprints and file checksums are recorded in `bundle.yaml`.", "",
            "## Community/support", "", "- [PRA documentation](https://einnovator.github.io/pdattention/)",
            "- [Source repository](https://github.com/einnovator/pdattention)", "- [Issues](https://github.com/einnovator/pdattention/issues)",
            "- [Contribution guide](https://github.com/einnovator/pdattention/blob/main/CONTRIBUTING.md)", "",
        ]
        if collection:
            lines[-1:-1] = [f"- [Canonical PRA Bundles Collection](https://huggingface.co/collections/{collection})"]
        if publisher:
            lines[-1:-1] = [f"- [{publisher} on Hugging Face](https://huggingface.co/{publisher})"]
        return "\n".join(lines)


def _metric(value: Any) -> str:
    if value is None or value == "":
        return "NEEDS_RUN"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _signed(value: Any) -> str:
    return "NEEDS_RUN" if value is None else f"{float(value):+.4f}"


def _signed_pct(value: Any) -> str:
    return "NEEDS_RUN" if value is None else f"{float(value):+.1f}%"


def _bytes(value: Any) -> str:
    if value is None:
        return "NEEDS_RUN"
    amount = float(value)
    unit = "B"
    for candidate in ("KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024:
            break
        amount /= 1024
        unit = candidate
    return f"{amount:.2f} {unit}"


def _public_mode(value: Any) -> str:
    return {"E0": "Selected Context", "E1": "Typed Transport", "E2": "Native Memory", "E3": "Native Serving", "E2_HOT": "Native Memory (hot)"}.get(str(value), str(value))


def _qualification_rows(bundle: PRAModelBundle) -> list[Mapping[str, Any]]:
    if not isinstance(bundle.qualification, Mapping):
        return []
    rows = bundle.qualification.get("metrics", bundle.qualification.get("rows", []))
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, Sequence) else []


def _canonical_evidence_rows(bundle: PRAModelBundle) -> list[CanonicalEvidenceRecord]:
    qualification = bundle.qualification if isinstance(bundle.qualification, Mapping) else {}
    raw = qualification.get("canonical_evidence", [])
    values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)) else [raw]
    records = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        fields = CanonicalEvidenceRecord.model_fields
        records.append(CanonicalEvidenceRecord.model_validate({name: value[name] for name in fields if name in value}))
    return records


def _condition_coverage(
    records: Sequence[CanonicalEvidenceRecord],
    condition: EvidenceCondition,
    fallback: str = "NEEDS_RUN",
) -> str:
    """Summarize measured scalar coverage for one card profile row."""

    observations = [
        observation
        for record in records
        for observation in record.conditions[condition].metrics.values()
    ]
    measured = sum(observation.state == MeasurementState.MEASURED for observation in observations)
    if measured:
        return (
            f"MEASURED ({measured})"
            if measured == len(observations)
            else f"PARTIAL ({measured}/{len(observations)})"
        )
    states = {observation.state.value for observation in observations}
    for state in (
        "BLOCKED",
        "NEEDS_RUN",
        "CALIBRATION_PENDING",
        "NO_QUALIFIED_ADAPTER",
        "NOT_APPLICABLE",
    ):
        if state in states:
            return state
    if "NOT_MEASURED" in states:
        return "NEEDS_RUN"
    return fallback


def validate_model_card(text: str, bundle: PRAModelBundle | None = None) -> dict[str, Any]:
    """Validate front matter and the public usage/evidence contract."""

    errors: list[str] = []
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        errors.append("model card requires YAML front matter")
        metadata: Mapping[str, Any] = {}
    else:
        metadata = yaml.safe_load(text.split("\n---\n", 1)[0][4:]) or {}
    if metadata.get("library_name") != "pra":
        errors.append("model card library_name must be 'pra'")
    if not metadata.get("base_model"):
        errors.append("model card base_model is required")
    tags = set(metadata.get("tags", []))
    if not {"pra", "progressive-retrieval-attention"}.issubset(tags):
        errors.append("model card requires PRA discoverability tags")
    for section in PUBLIC_CARD_SECTIONS:
        if f"## {section}" not in text:
            errors.append(f"model card missing section: {section}")
    prose = text.split("\n---\n", 1)[-1]
    if any(internal in prose for internal in (" E0 ", " E1 ", " E2 ", " E3 ")):
        errors.append("model card exposes paper-only execution labels")
    if bundle is not None and metadata.get("base_model") != bundle.base_model.get("id"):
        errors.append("model card base_model disagrees with bundle")
    if errors:
        raise BundleValidationError("Invalid model card: " + "; ".join(errors))
    return dict(metadata)


@dataclass(frozen=True)
class BundleRegistryEntry:
    """One immutable trusted-registry candidate."""

    name: str
    base_model: str
    base_revision: str
    architecture: str
    bundle_repo: str
    bundle_revision: str
    pra_version: str
    schema_version: int
    trust: str
    engine_compatibility: Mapping[str, Any]
    profiles: Sequence[str]
    qualification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "base_model": self.base_model, "base_revision": self.base_revision,
            "architecture": self.architecture, "bundle_repo": self.bundle_repo,
            "bundle_revision": self.bundle_revision, "pra_version": self.pra_version,
            "schema_version": self.schema_version, "trust": self.trust,
            "engine_compatibility": dict(self.engine_compatibility), "profiles": list(self.profiles),
            "qualification": self.qualification,
        }


class TrustedBundleRegistry:
    """Declarative allow-list used only for automatic bundle selection."""

    def __init__(self, entries: Iterable[BundleRegistryEntry]) -> None:
        self.entries = tuple(entries)

    @classmethod
    def default(cls) -> "TrustedBundleRegistry":
        path = Path(__file__).with_name("model_profiles") / "bundle_registry.yaml"
        if not path.is_file():
            return cls(())
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(BundleRegistryEntry(**entry) for entry in value.get("bundles", ()))

    def list(
        self,
        *,
        model: str | None = None,
        family: str | None = None,
        engine: str | None = None,
        qualification: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return deterministic registry rows matching product-facing filters."""

        rows = []
        for entry in self.entries:
            if model and entry.base_model.lower() != model.lower():
                continue
            if family and family.lower() not in entry.architecture.lower() and family.lower() not in entry.base_model.lower():
                continue
            if engine and entry.engine_compatibility.get(engine.lower()) in {None, "unsupported", False}:
                continue
            if qualification and entry.qualification.lower() != qualification.lower():
                continue
            if query:
                searchable = " ".join(
                    (
                        entry.name,
                        entry.base_model,
                        entry.architecture,
                        entry.bundle_repo,
                        entry.qualification,
                        *entry.profiles,
                    )
                ).lower()
                if query.lower() not in searchable:
                    continue
            rows.append(entry.to_dict())
        return sorted(rows, key=lambda row: str(row["bundle_repo"]).lower())

    def candidates(self, model: str, *, revision: str | None = None, engine: str | None = None) -> list[BundleRegistryEntry]:
        rows = []
        for entry in self.entries:
            if entry.base_model.lower() != model.lower() or (revision and entry.base_revision != revision):
                continue
            if engine and entry.engine_compatibility.get(engine) in {None, "unsupported", False}:
                continue
            if entry.trust == "eInnovator-qualified":
                rows.append(entry)
        return rows


@dataclass(frozen=True)
class BundleAvailability:
    """Download-free result for a trusted published bundle lookup."""

    status: str
    source: str | None
    bundle_revision: str | None
    base_revision: str | None
    compatibility: str
    trust: str
    qualification: str | None
    profiles: Sequence[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "bundle_revision": self.bundle_revision,
            "base_revision": self.base_revision,
            "compatibility": self.compatibility,
            "trust": self.trust,
            "qualification": self.qualification,
            "profiles": list(self.profiles),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BundleResolution:
    requested: str
    status: str
    source: str | None
    resolved_revision: str | None
    local_path: str | None
    trust: str
    reason: str
    bundle: PRAModelBundle | None = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        value = {"requested": self.requested, "status": self.status, "source": self.source,
                 "resolved_revision": self.resolved_revision, "local_path": self.local_path,
                 "trust": self.trust, "reason": self.reason}
        if self.bundle is not None:
            value["bundle"] = self.bundle.inspect()
        return value


class BundleResolver:
    """Resolve none, local, explicit Hub, or trusted automatic bundles."""

    def __init__(self, registry: TrustedBundleRegistry | None = None) -> None:
        self.registry = registry or TrustedBundleRegistry.default()

    def discover(
        self,
        *,
        model: str,
        model_revision: str | None = None,
        engine: str | None = None,
    ) -> BundleAvailability:
        """Find a trusted registry entry without downloading its bundle payload."""

        candidates = self.registry.candidates(model, engine=engine)
        if not candidates:
            return BundleAvailability(
                "NO_TRUSTED_MATCH", None, None, None, "none", "none", None, (),
                "No trusted published bundle matches the model and engine.",
            )
        if model_revision:
            exact = [entry for entry in candidates if entry.base_revision == model_revision]
            if not exact:
                entry = candidates[0]
                return BundleAvailability(
                    "INCOMPATIBLE", entry.bundle_repo, entry.bundle_revision,
                    entry.base_revision, "base-revision-mismatch", entry.trust,
                    entry.qualification, entry.profiles,
                    "A trusted bundle exists for this model, but its base revision does not "
                    "match the inspected model revision.",
                )
            entry = exact[0]
            compatibility = "exact"
            reason = "Trusted model identity, immutable base revision, and engine compatibility match."
        else:
            entry = candidates[0]
            compatibility = "model-id-only"
            reason = "Trusted model identity and engine compatibility match; base revision is unresolved."
        return BundleAvailability(
            "FOUND", entry.bundle_repo, entry.bundle_revision, entry.base_revision,
            compatibility, entry.trust, entry.qualification, entry.profiles, reason,
        )

    def resolve(self, requested: str | None, *, model: str, model_revision: str | None = None, engine: str | None = None) -> BundleResolution:
        value = requested or "auto"
        if value.lower() == "none":
            return BundleResolution(value, "DISABLED", None, None, None, "none", "Bundle-specific adapters explicitly disabled; PRA remains available.")
        if value.lower() == "auto":
            candidates = self.registry.candidates(model, revision=model_revision, engine=engine)
            if not candidates:
                availability = self.discover(
                    model=model, model_revision=model_revision, engine=engine
                )
                return BundleResolution(
                    value, availability.status, availability.source, None, None,
                    availability.trust, availability.reason,
                )
            entry = candidates[0]
            source, revision, trust = entry.bundle_repo, entry.bundle_revision, entry.trust
            reason = "Matched trusted base-model identity, engine compatibility, and qualification metadata."
        else:
            source, revision = value, None
            trust = "local/private" if Path(value).expanduser().exists() else "community"
            reason = "Explicit bundle source selected by the user."
        bundle = PRAModelBundle.from_pretrained(source, revision=revision)
        actual_model = str(bundle.base_model.get("id", ""))
        if actual_model.lower() != model.lower():
            raise BundleValidationError(f"Bundle base model {actual_model!r} does not match requested model {model!r}.")
        if model_revision and bundle.base_model.get("revision") != model_revision:
            raise BundleValidationError("Bundle base revision does not match the requested model revision.")
        return BundleResolution(value, "RESOLVED", source, bundle.resolved_revision or revision,
                                str(bundle.local_path) if bundle.local_path else None, trust, reason, bundle)


class HubBundleCatalog:
    """Read-only discovery of PRA bundles published on Hugging Face Hub.

    Hub search is deliberately separate from ``TrustedBundleRegistry``. A live
    result is descriptive only; automatic resolution still requires a pinned
    registry entry with bounded qualification metadata.
    """

    def __init__(
        self,
        api: Any | None = None,
        registry: TrustedBundleRegistry | None = None,
    ) -> None:
        self._api = api
        self.registry = registry or TrustedBundleRegistry.default()

    def search(
        self,
        query: str = "pra",
        *,
        author: str | None = "EInnovator",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search Hub metadata without downloading bundle payloads."""

        api = self._api
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise ImportError(
                    "Hub bundle search requires the 'hf-hub' optional dependency."
                ) from error
            api = HfApi()

        trusted = {
            entry.bundle_repo.lower(): entry for entry in self.registry.entries
        }
        fetch_limit = max(limit * 4, 50)
        models = api.list_models(
            author=author,
            search=query or "pra",
            full=True,
            cardData=True,
            sort="lastModified",
            direction=-1,
            limit=fetch_limit,
        )
        rows: list[dict[str, Any]] = []
        for model in models:
            repo_id = str(
                getattr(model, "id", None) or getattr(model, "modelId", "")
            )
            if not repo_id:
                continue
            tags = [str(tag) for tag in (getattr(model, "tags", None) or ())]
            card = self._card_data(getattr(model, "cardData", None))
            library_name = str(card.get("library_name", ""))
            repository_name = repo_id.rsplit("/", 1)[-1].lower()
            tag_names = {tag.lower() for tag in tags}
            if not (
                repository_name.startswith("pra-")
                or library_name.lower() == "pra"
                or "pra" in tag_names
                or "progressive-retrieval-attention" in tag_names
            ):
                continue

            entry = trusted.get(repo_id.lower())
            last_modified = getattr(model, "lastModified", None)
            if hasattr(last_modified, "isoformat"):
                last_modified = last_modified.isoformat()
            base_model = entry.base_model if entry else card.get("base_model")
            rows.append(
                {
                    "repo_id": repo_id,
                    "url": f"https://huggingface.co/{repo_id}",
                    "base_model": base_model,
                    "hub_revision": getattr(model, "sha", None),
                    "registry_revision": entry.bundle_revision if entry else None,
                    "qualification": entry.qualification if entry else "UNREGISTERED",
                    "trust": entry.trust if entry else "hub-discovered",
                    "auto_resolvable": (
                        entry is not None and entry.trust == "eInnovator-qualified"
                    ),
                    "profiles": list(entry.profiles) if entry else [],
                    "downloads": getattr(model, "downloads", None),
                    "likes": getattr(model, "likes", None),
                    "last_modified": last_modified,
                    "private": bool(getattr(model, "private", False)),
                    "gated": getattr(model, "gated", False),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _card_data(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            result = to_dict()
            return result if isinstance(result, Mapping) else {}
        return {}


class HubPublisher:
    """Validated Hugging Face Hub transport and repeatable batch publisher."""

    def push(
        self, bundle: str | Path, repo_id: str, *, revision: str | None = None,
        dry_run: bool = False, private: bool = False, collection: str | None = None,
        license_name: str | None = None, commit_message: str = "Publish PRA model bundle",
        tag: str | None = None,
    ) -> dict[str, Any]:
        source = Path(bundle).resolve()
        loaded = PRAModelBundle.from_pretrained(source)
        if license_name and loaded.provenance.get("license") not in {None, license_name}:
            raise BundleValidationError("CLI license disagrees with bundle provenance.")
        files = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file())
        result: dict[str, Any] = {
            "repo_id": repo_id, "requested_revision": revision, "files": files,
            "dry_run": dry_run, "base_model": dict(loaded.base_model),
            "schema_version": loaded.schema_version, "collection": collection,
        }
        if dry_run:
            return result
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
        except ImportError as error:
            raise ImportError("Publishing requires the 'hf-hub' optional dependency.") from error
        api = HfApi()
        try:
            remote_manifest = yaml.safe_load(
                Path(hf_hub_download(repo_id, "bundle.yaml", revision=revision)).read_text(
                    encoding="utf-8"
                )
            ) or {}
            remote_model = remote_manifest.get("base_model", {})
            if remote_model.get("id") != loaded.base_model.get("id") or remote_model.get("revision") != loaded.base_model.get("revision"):
                raise BundleValidationError("Existing Hub repository targets an incompatible base model or revision.")
        except (RepositoryNotFoundError, HfHubHTTPError):
            pass
        api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
        commit = api.upload_folder(repo_id=repo_id, folder_path=source, revision=revision,
                                   commit_message=commit_message, repo_type="model")
        commit_oid = getattr(commit, "oid", None)
        if tag:
            api.create_tag(repo_id, tag=tag, revision=commit_oid or revision, exist_ok=True)
        collection_slug = self._add_to_collection(api, collection, repo_id) if collection else None
        result.update(dry_run=False, commit=commit_oid,
                      url=getattr(commit, "repo_url", f"https://huggingface.co/{repo_id}"),
                      collection=collection_slug, tag=tag)
        return result

    @staticmethod
    def _add_to_collection(api: Any, collection: str, repo_id: str) -> str:
        try:
            slug = api.get_collection(collection).slug
        except Exception:
            namespace, _, title = collection.partition("/")
            created = api.create_collection(
                title=(title or "Progressive Retrieval Attention PRA Bundles").replace("-", " ").title(),
                namespace=namespace or None,
                description="Qualified and research Progressive Retrieval Attention model bundles.",
                exists_ok=True,
            )
            slug = created.slug
        api.add_collection_item(slug, item_id=repo_id, item_type="model", exists_ok=True)
        return slug

    def pull(self, repo_id: str, output: str | Path | None = None, *, revision: str | None = None) -> dict[str, Any]:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as error:
            raise ImportError("Hub pull requires the 'hf-hub' optional dependency.") from error
        path = Path(snapshot_download(repo_id, revision=revision, local_dir=output))
        resolved = _snapshot_revision(path) or HfApi().model_info(repo_id, revision=revision).sha
        bundle = PRAModelBundle.from_pretrained(path)
        return {"repo_id": repo_id, "requested_revision": revision, "resolved_revision": resolved,
                "cache_path": str(path), "schema_version": bundle.schema_version,
                "base_model": dict(bundle.base_model),
                "validation_status": bundle.qualification.get("status", "NOT_MEASURED")}

    def publish_manifest(self, manifest: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
        """Publish an idempotent list of bundles with per-item result logging."""

        path = Path(manifest)
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        results: list[dict[str, Any]] = []
        for item in value.get("bundles", []):
            try:
                result = self.push(item["bundle"], item["repo_id"], revision=item.get("revision"),
                                   collection=item.get("collection"), private=bool(item.get("private", False)),
                                   commit_message=item.get("commit_message", "Publish PRA model bundle"),
                                   tag=item.get("tag"), dry_run=dry_run)
                result["status"] = "VALIDATED" if dry_run else "PUBLISHED"
            except Exception as error:
                result = {"repo_id": item.get("repo_id"), "status": "FAILED", "error": f"{type(error).__name__}: {error}"}
            results.append(result)
        return {"manifest": str(path), "dry_run": dry_run, "results": results}
