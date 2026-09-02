"""Generate the public CLI reference directly from the Click command tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import click

from pra_hf.cli import cli


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/site/cli-reference.md"


EXAMPLES = {
    "pra gateway serve": "pra gateway serve --mode selected-context --backend vllm --backend-url http://127.0.0.1:8000/v1 --sessions-dir .pra/gateway-sessions",
    "pra engine connect": "pra engine connect http://127.0.0.1:9101 --name local-vllm --token-env PRA_MANAGEMENT_TOKEN",
    "pra engine health": "pra engine health local-vllm",
    "pra engine config": "pra engine config local-vllm --yaml",
    "pra engine storage": "pra engine storage local-vllm --json",
    "pra engine sessions": "pra engine sessions local-vllm --json",
    "pra engine resources": "pra engine resources local-vllm --json",
    "pra engine models": "pra engine models local-vllm",
    "pra engine profiles": "pra engine profiles local-vllm",
    "pra engine capabilities": "pra engine capabilities local-vllm",
    "pra engine audit": "pra engine audit local-vllm --json",
    "pra engine inspect": "pra engine inspect --management-url http://127.0.0.1:9101",
    "pra engine patch-config": "pra engine patch-config local-vllm --patch profile-patch.yaml",
    "pra engine action": "pra engine action promote local-vllm --resource-id RESOURCE_ID --idempotency-key promote-42",
    "pra engine serve": "pra engine serve --engine vllm --model Qwen/Qwen3-4B --inference-url http://127.0.0.1:8000 --port 9101",
    "pra doctor": "pra doctor",
    "pra engines": "pra engines --details mlx",
    "pra inspect": "pra inspect Qwen/Qwen3-0.6B --engine hf\npra inspect Qwen/Qwen3-0.6B --engine hf --pra-bundle auto",
    "pra evaluate": "pra evaluate Qwen/Qwen3-1.7B --engine hf --dataset qasper --pra-bundle auto --measurements results.json -o .pra/runs/qasper",
    "pra recommend": "pra recommend .pra/runs/qasper",
    "pra report": "pra report .pra/runs/qasper --format html -o .pra/reports/qasper.html",
    "pra qualify native-memory": "pra qualify native-memory Qwen/Qwen3-1.7B --engine hf --dataset qasper --measurements results.json -o .pra/runs/native-memory",
    "pra qualify native-serving": "pra qualify native-serving Qwen/Qwen3-1.7B --engine vllm --dataset qasper --measurements results.json -o .pra/runs/native-serving",
    "pra assess init": "pra assess init customer-workload",
    "pra assess run": "pra assess run .pra/assessments/customer-workload --measurements results.json",
    "pra assess report": "pra assess report .pra/assessments/customer-workload --format html",
    "pra model inspect": "pra model inspect Qwen/Qwen3-1.7B --validate",
    "pra model adapt": "pra model adapt Qwen/Qwen3-1.7B -o .pra/adapters/qwen3",
    "pra model validate": "pra model validate Qwen/Qwen3-1.7B --adapter .pra/adapters/qwen3 --suite standard -o .pra/runs/model-validation",
    "pra model onboard": "pra model onboard Qwen/Qwen3-1.7B --suite standard --engine hf -o .pra/runs/onboarding",
    "pra adapter inspect": "pra adapter inspect .pra/adapters/router",
    "pra adapter train routing": "pra adapter train routing Qwen/Qwen3-1.7B --model-family qwen --train-features train.jsonl --validation-features valid.jsonl -D qasper -o .pra/adapters/router",
    "pra adapter train memory": "pra adapter train memory",
    "pra adapter train late-band": "pra adapter train late-band",
    "pra adapter eval": "pra adapter eval .pra/adapters/router --features test.jsonl --query-strategy last",
    "pra profiles show": "pra profiles show Qwen/Qwen3-1.7B --workload qasper",
    "pra profiles calibrate": "pra profiles calibrate Qwen/Qwen3-1.7B --suite standard --engine hf --workload qasper -o .pra/runs/profile-calibration",
    "pra profiles compare": "pra profiles compare Qwen/Qwen3-1.7B --workload qasper",
    "pra profiles report": "pra profiles report Qwen/Qwen3-1.7B -o .pra/reports/profiles.md",
    "pra bundle build": "pra bundle build .pra/runs/profile-calibration -o .pra/bundles/qwen3",
    "pra bundle inspect": "pra bundle inspect .pra/bundles/qwen3",
    "pra bundle validate": "pra bundle validate .pra/bundles/qwen3",
    "pra bundle card": "pra bundle card .pra/bundles/qwen3 --update",
    "pra bundle list": "pra bundle list --model Qwen/Qwen3-0.6B",
    "pra bundle resolve": "pra bundle resolve Qwen/Qwen3-0.6B -e hf -a auto",
    "pra hf login": "pra hf login --check\npra hf login",
    "pra hf list": "pra hf list --family qwen --engine mlx",
    "pra hf search": "pra hf search qwen --author EInnovator --limit 20",
    "pra hf pull": "pra hf pull owner/pra-qwen3-0.6b --revision IMMUTABLE_COMMIT",
    "pra hf push": "pra hf push .pra/bundles/qwen3 owner/pra-qwen3-0.6b --collection owner/pra-bundles --tag v0.2.0rc1 --dry-run",
    "pra hf publish-manifest": "pra hf publish-manifest releases/pra_bundles.yaml --dry-run",
    "pra hf inspect": "pra hf inspect owner/Qwen3-PRA",
    "pra runtime init": "pra runtime init .pra/runtime --storage balanced --max-native-index-tokens 32768",
    "pra runtime serve": "pra runtime serve Qwen/Qwen3-1.7B --engine hf -m auto --explain --profile recommended --port 8000",
    "pra runtime inspect": "pra runtime inspect Qwen/Qwen3-1.7B --engine hf --storage balanced",
    "pra runtime doctor": "pra runtime doctor --engine hf",
    "pra runtime benchmark": "pra runtime benchmark Qwen/Qwen3-1.7B --engine hf --profile BALANCED --storage balanced -o .pra/benchmarks/qwen3",
    "pra runtime capabilities": "pra runtime capabilities --json",
    "pra serve": "pra serve Qwen/Qwen3-1.7B --engine hf -m auto --explain --profile recommended --port 8000",
    "pra agent chat": "pra agent chat --model Qwen/Qwen3-0.6B --workspace . --task \"Inspect this repository\"",
    "pra agent run": "pra agent run --profile work --session issue-42 \"Summarize the current task state.\" --json",
    "pra agent inspect": "pra agent inspect --profile work --yaml",
    "pra agent start": "pra agent start --profile work --host 127.0.0.1 --port 8765 --detach --open",
    "pra agent stop": "pra agent stop",
}


OUTPUTS = {
    "pra gateway serve": "PRA gateway on http://127.0.0.1:8080 -> vllm\nSelected Context: enabled\nTyped resource transport: disabled\nEffective mode: Selected Context",
    "pra engine connect": "Name: local-vllm\nURL: http://127.0.0.1:9101\nProtocol: pra-management/1\nStored secret: false",
    "pra engine health": "Status: healthy\nProtocol: pra-management/1\nInstance ID: 2f9c...",
    "pra engine config": "effective:\n  profile: BALANCED\nobserved_revision: 1\nin_sync: true",
    "pra engine storage": "{\n  \"tiers\": {\"hot\": {\"bytes\": 1048576, \"resources\": 2}},\n  \"maintenance_status\": \"running\"\n}",
    "pra engine sessions": "{\"items\": [{\"session_id\": \"8e0f...\", \"status\": \"active\"}], \"total\": 1}",
    "pra engine resources": "{\"items\": [{\"resource_id\": \"5a31...\", \"storage_tier\": \"hot\"}], \"total\": 1}",
    "pra engine models": "Items:\n  Model Id: Qwen/Qwen3-4B\nTotal: 1",
    "pra engine profiles": "Items:\n  Name: BALANCED\n  Qualification Status: VALIDATED",
    "pra engine capabilities": "Selected Context: AVAILABLE\nNative Memory: CANDIDATE\nManagement API Version: pra-management/1",
    "pra engine audit": "{\"items\": [{\"event\": \"RESOURCE_PROMOTED\", \"result\": \"success\"}]}",
    "pra engine inspect": "Info:\n  Engine: vllm\nCapabilities:\n  Management Api Version: pra-management/1\nState:\n  In Sync: true",
    "pra engine patch-config": "Observed Revision: 2\nIn Sync: true\nEffective:\n  Profile: ECONOMY",
    "pra engine action": "Action: promote\nStatus: success\nResource Id: RESOURCE_ID\nIdempotent Replay: false",
    "pra engine serve": "PRA management API (vllm) on http://127.0.0.1:9101\nOpenAPI: http://127.0.0.1:9101/openapi.json\nSwagger: http://127.0.0.1:9101/docs",
    "pra doctor": "System: Python 3.10.x\nTorch: AVAILABLE\nDevice backend: CPU\nNext action: pra inspect MODEL --engine ENGINE",
    "pra engines": "Engine: MLX\nSelected Context: available\nNative Memory: measured\nRecommended today: use the qualified profile for this model and hardware",
    "pra inspect": "Model: Qwen/Qwen3-0.6B\nRevision: c1899de...\nEngine: hf\n\nPublished PRA bundle found\n  Repository: EInnovator/pra-qwen3-0.6b\n  Revision: 25e6907...\n  Base revision: c1899de...\n  Compatibility: exact\n  Trust: eInnovator-qualified\n\nWith --pra-bundle auto:\nPRA bundle resolution\n  Status: RESOLVED\n  Compatibility: exact",
    "pra evaluate": "Run: .pra/runs/qasper\nModes: full_context, selected_context\nMeasurements imported: results.json\nRecommendation status: PENDING",
    "pra recommend": "Recommended mode: selected_context\nReason: quality gate passed; native economics not qualified",
    "pra report": ".pra/reports/qasper.html",
    "pra qualify native-memory": "Qualification: native_memory\nStatus: PASS\nRun: .pra/runs/native-memory",
    "pra qualify native-serving": "Qualification: native_serving\nStatus: PENDING\nMissing: concurrent scheduler measurements",
    "pra assess init": ".pra/assessments/customer-workload/config.yaml",
    "pra assess run": "Assessment: customer-workload\nStatus: complete\nReport data: .pra/assessments/customer-workload/run",
    "pra assess report": ".pra/assessments/customer-workload/report.html",
    "pra model inspect": "Family: qwen\nStructural mapping: built-in\nValidation requested: true\nStatus: candidate until validation completes",
    "pra model adapt": "Adapter: .pra/adapters/qwen3/pra_adapter.yaml\nFamily: qwen\nLearned weights: none",
    "pra model validate": "Suite: standard\nDisabled-PRA parity: PASS\nNative K/V capture: PASS\nGeneration: PASS",
    "pra model onboard": "Runs: 1\nModel: Qwen/Qwen3-1.7B\nOutput: .pra/runs/onboarding/Qwen--Qwen3-1.7B",
    "pra adapter inspect": "Adapter type: routing\nBase model: Qwen/Qwen3-1.7B\nRouting dimension: 128",
    "pra adapter train routing": "Output: .pra/adapters/router\nSteps: 512\nValidation metrics: .pra/adapters/router/metrics.json",
    "pra adapter train memory": "Error: Memory-adapter training remains research-only; no certified dataset pipeline is packaged yet.",
    "pra adapter train late-band": "Error: Late-band LoRA remains research-only and is not a certified product path.",
    "pra adapter eval": "Examples: 120\nTop-k recall: 0.81\nMean reciprocal rank: 0.72",
    "pra profiles show": "Model: Qwen/Qwen3-1.7B\nWorkload: qasper\nProfiles: REFERENCE_CORRECTNESS, BALANCED",
    "pra profiles calibrate": "Output: .pra/runs/profile-calibration\nEvidence tier: measured\nRecommended profile: BALANCED",
    "pra profiles compare": "REFERENCE_CORRECTNESS: quality 1.000\nBALANCED: quality 0.998\nReduced candidates: calibration pending",
    "pra profiles report": ".pra/reports/profiles.md",
    "pra bundle build": "Output: .pra/bundles/qwen3\nBase model: Qwen/Qwen3-1.7B\nBundle schema: 2",
    "pra bundle inspect": "Base model: Qwen/Qwen3-1.7B\nProfiles: BALANCED\nEvidence artifacts: present",
    "pra bundle validate": "Status: VALID\nModel: Qwen/Qwen3-1.7B\nSchema version: 2\nChecksums: verified",
    "pra bundle card": ".pra/bundles/qwen3/README.md",
    "pra bundle list": "Bundles: 1\nQwen/Qwen3-0.6B -> owner/pra-qwen3-0.6b\nTrust: eInnovator-qualified",
    "pra bundle resolve": "Status: RESOLVED\nRevision: IMMUTABLE_COMMIT\nTrust: eInnovator-qualified\nCache: HF snapshot cache",
    "pra hf login": "Status: AUTHENTICATED\nName: maintainer\nOrganizations: EInnovator",
    "pra hf list": "PRA bundle catalog (3)\nSource: trusted-registry\n\nEInnovator/pra-qwen3-0.6b\n  Base model: Qwen/Qwen3-0.6B\n  Qualification: CONTROLLED\n  Trust: eInnovator-qualified",
    "pra hf search": "PRA bundle catalog (3)\nSource: hugging-face-hub\n\nEInnovator/pra-qwen3-0.6b\n  Base model: Qwen/Qwen3-0.6B\n  Qualification: CONTROLLED\n  Trust: eInnovator-qualified\n  Auto resolvable: True",
    "pra hf pull": "Repository: owner/pra-qwen3-0.6b\nResolved revision: IMMUTABLE_COMMIT\nCache path: HF snapshot cache\nStatus: VALID",
    "pra hf push": "Dry run: true\nRepository: owner/Qwen3-PRA\nFiles checked: 8\nUploaded: 0",
    "pra hf inspect": "Source: owner/Qwen3-PRA\nBase model: Qwen/Qwen3-1.7B\nBundle schema: 1",
    "pra hf publish-manifest": "Manifest: releases/pra_bundles.yaml\nValidated: 1\nUploaded: 0\nDry run: true",
    "pra runtime init": ".pra/runtime/pra.yaml\nStorage profile: balanced\nNative index budget: 32768 tokens",
    "pra runtime serve": "Runtime: hf\nStatus: healthy\nRequested mode: auto\nResolved mode: selected-context\nReason: native economics require qualified evidence\nEndpoint: http://127.0.0.1:8000",
    "pra runtime inspect": "Engine: hf\nStorage: balanced\nProfile: provider default\nEndpoint: embedded",
    "pra runtime doctor": "Engine: hf\nProvider: AVAILABLE\nModel endpoint: not requested\nNext action: pra runtime inspect MODEL --engine hf",
    "pra runtime benchmark": "Output: .pra/benchmarks/qwen3\nProfile: BALANCED\nMetrics: metrics.json",
    "pra runtime capabilities": "{\n  \"typed_records\": true,\n  \"native_memory\": \"engine-dependent\",\n  \"streaming\": true\n}",
    "pra serve": "Status: healthy\nRequested mode: auto\nResolved mode: selected-context\nResolution reason: native economics require qualified evidence",
    "pra agent chat": "Agent profile: default\nRuntime: embedded/hf\nSession: new\n> Inspect this repository",
    "pra agent run": "{\n  \"response\": \"The current task is ...\",\n  \"session_id\": \"issue-42\",\n  \"tool_calls\": 0\n}",
    "pra agent inspect": "agent_profile: work\nmodel: Qwen/Qwen3-0.6B\ncontext_transport: auto\ntools: ask",
    "pra agent start": "PRA Agent Web UI started\nURL: http://127.0.0.1:8765\nDetached: true",
    "pra agent stop": "PRA Agent Web UI stopped.",
}


OPTION_HELP = {
    "host": "Bind address for the local service.",
    "port": "TCP port for the local service.",
    "mode": "Select the product execution or gateway mediation mode.",
    "backend": "Select the downstream gateway adapter.",
    "backend_url": "Base URL of the existing downstream model endpoint.",
    "model": "Model identifier or local model path.",
    "prefix_cache_mode": "Declare or auto-detect downstream prefix-cache behavior.",
    "session_state": "Enable or disable downstream session state.",
    "incremental_messages": "Send message deltas when supported, or full history.",
    "resource_delta": "Send resource operations when supported, or full inventories.",
    "cache_affinity": "Enable or disable stable cache-affinity hints.",
    "fallback_injection": "Choose where Selected Context is inserted into ordinary messages.",
    "sessions_dir": "Persist gateway session metadata under this directory.",
    "verbose": "Show additional resolution and diagnostic detail.",
    "json_output": "Emit machine-readable JSON.",
    "yaml_output": "Emit machine-readable YAML.",
    "details": "Show the detailed record for one engine.",
    "engine": "Select the runtime or evidence-registry engine.",
    "revision": "Pin a model, bundle, or Hub revision.",
    "dataset": "Name the evaluation dataset.",
    "datasets": "Record one or more training datasets; repeat the option.",
    "measurements": "Import measured mode results from JSON.",
    "include_native_memory": "Include Native Memory in the evaluation candidate set.",
    "include_native_serving": "Include Native Serving in the evaluation candidate set.",
    "quality_threshold": "Minimum retained-quality ratio required by the gate.",
    "profile": "Select a named PRA or agent profile.",
    "output": "Write artifacts to this file or directory.",
    "format_name": "Choose the report output format.",
    "root": "Root directory for assessment workspaces.",
    "validate": "Run structural validation during inspection.",
    "device": "Execution device such as auto, cpu, cuda, or mps.",
    "force": "Replace or rerun artifacts that already exist.",
    "adapter": "Path or identifier of a structural adapter.",
    "suite": "Select the validation or calibration suite.",
    "manifest": "Read a multi-model onboarding manifest.",
    "jobs": "Maximum number of onboarding jobs.",
    "validation_datasets": "Record one or more validation datasets; repeat the option.",
    "train_features": "Cached training feature file; repeat for multiple shards.",
    "validation_features": "Cached validation feature file; repeat for multiple shards.",
    "model_family": "Select the structural model-family mapping.",
    "routing_dim": "Width of the learned routing projection.",
    "steps": "Number of adapter optimization steps.",
    "seed": "Random seed used by adapter training.",
    "features": "Feature file used for evaluation; repeat for multiple shards.",
    "query_strategy": "Choose how evaluation derives its routing query.",
    "workload": "Filter or label profile evidence by workload.",
    "registry": "Use an alternate profile benchmark registry.",
    "dry_run": "Validate publication without uploading files.",
    "yes": "Skip the interactive publication confirmation.",
    "update": "Write the regenerated model card into the bundle.",
    "family": "Filter trusted bundles by model family.",
    "query": "Filter registry metadata or search Hugging Face Hub.",
    "qualification": "Require an exact bundle qualification tier.",
    "author": "Limit Hub results to one organization or user namespace.",
    "all_authors": "Search every Hub namespace; discovered results remain untrusted unless registered.",
    "limit": "Maximum number of matching Hub bundles to return.",
    "private": "Create a private Hub repository; public is the default.",
    "collection": "Add the published bundle to this Hugging Face Collection.",
    "license_name": "Assert a license only when it matches bundle provenance.",
    "commit_message": "Set the Hugging Face upload commit message.",
    "tag": "Create an immutable Hub release tag after upload.",
    "check": "Check existing authentication without opening an interactive prompt.",
    "max_native_index_tokens": "Set the native-index ingestion token budget.",
    "max_native_index_bytes": "Set the native-index ingestion byte budget.",
    "defer_native_index": "Build native selected-region state lazily.",
    "storage": "Select the semantic storage lifecycle profile.",
    "endpoint": "Use a remote runtime or gateway endpoint.",
    "pra_bundle": "Load a PRA bundle or configuration override.",
    "storage_config": "Load a detailed storage policy file.",
    "engine_args": "Pass a provider-specific engine argument; repeat as needed.",
    "pra": "Override the PRA profile, bundle, or configuration for the agent.",
    "config": "Load an explicit agent profile document.",
    "workspace": "Set the agent workspace directory.",
    "skills": "Discover skills under this directory; repeat as needed.",
    "context_transport": "Require typed PRA, require text, or negotiate automatically.",
    "allow_text_fallback": "Allow or reject explicit Selected Context fallback.",
    "session": "Use this agent session identifier.",
    "resume": "Resume persisted session state.",
    "task": "Set or update the active task description.",
    "detach": "Run the Web UI as a detached process.",
    "open_browser": "Open the Web UI in the default browser.",
    "management_url": "Use a one-off PRA management API URL.",
    "token": "Authenticate with a bearer token; prefer PRA_MANAGEMENT_TOKEN.",
    "token_env": "Name the environment variable containing a bearer token.",
    "auth_mode": "Select local management authentication.",
    "inference_url": "Identify the engine inference endpoint managed by the sidecar.",
    "engine_version": "Report the observed engine version when it cannot be discovered.",
    "metrics_url": "Publish the engine Prometheus endpoint link.",
    "trace_backend_url": "Publish the configured trace-backend link.",
    "grafana_url": "Publish the configured Grafana link.",
    "tls_certfile": "Serve HTTPS with this certificate chain.",
    "tls_keyfile": "Serve HTTPS with this private-key file.",
    "tls_ca_certs": "Require client certificates issued by this CA bundle.",
    "patch_path": "Read a bounded YAML or JSON configuration patch.",
    "resource_id": "Address a privacy-safe resource identifier returned by the API.",
    "tenant_id": "Authorize the storage action for this tenant.",
    "idempotency_key": "Deduplicate a retried management action.",
}


ARGUMENT_HELP = {
    "model": "Model identifier or local model path.",
    "run": "Qualification or calibration run directory.",
    "name": "New assessment name.",
    "assessment": "Assessment directory created by `pra assess init`.",
    "source": "Local artifact path or supported remote identifier.",
    "bundle": "Local PRA bundle directory.",
    "repo_id": "Hugging Face repository identifier.",
    "query": "Optional Hugging Face search text; defaults to `pra`.",
    "directory": "Runtime configuration directory to create.",
    "legacy_model": "Optional compatibility spelling for the model; prefer `--model`.",
    "prompt": "One noninteractive agent instruction; stdin is used when omitted.",
    "url": "PRA management API base URL.",
    "target": "Saved connection name or management API URL.",
    "action": "Bounded management action name.",
}


GROUPS = (
    ("Gateway", ("pra gateway",)),
    ("Engine management", ("pra engine",)),
    ("Environment and qualification", ("pra doctor", "pra engines", "pra inspect", "pra evaluate", "pra recommend", "pra report", "pra qualify")),
    ("Assessments", ("pra assess",)),
    ("Models", ("pra model",)),
    ("Learned adapters", ("pra adapter",)),
    ("Profiles", ("pra profiles",)),
    ("Bundles", ("pra bundle",)),
    ("Hugging Face Hub", ("pra hf",)),
    ("Runtime and serving", ("pra runtime", "pra serve")),
    ("Agents", ("pra agent",)),
)


def public_commands() -> dict[str, click.Command]:
    """Return every non-hidden leaf command under its complete invocation."""

    result: dict[str, click.Command] = {}

    def visit(group: click.Group, path: tuple[str, ...]) -> None:
        for name, command in group.commands.items():
            if command.hidden:
                continue
            current = (*path, name)
            if isinstance(command, click.Group):
                visit(command, current)
            else:
                result[" ".join(current)] = command

    visit(cli, ("pra",))
    return result


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _default(option: click.Option) -> str:
    if option.default is None or option.default == ():
        return "-"
    if option.is_flag:
        return "on" if option.default else "off"
    if isinstance(option.default, Path):
        return option.default.as_posix()
    return _escape(option.default)


def _value(option: click.Option) -> str:
    if option.is_flag:
        return "flag"
    if isinstance(option.type, click.Choice):
        value = " / ".join(str(choice) for choice in option.type.choices)
    elif isinstance(option.type, click.Path):
        value = "PATH"
    elif isinstance(option.type, click.IntRange):
        value = "INTEGER"
        if option.type.min is not None:
            value += f" >= {option.type.min}"
        if option.type.max is not None:
            value += f" <= {option.type.max}"
    elif isinstance(option.type, click.FloatRange):
        value = "FLOAT"
        if option.type.min is not None:
            value += f" >= {option.type.min}"
        if option.type.max is not None:
            value += f" <= {option.type.max}"
    else:
        value = option.type.name.upper()
    if option.multiple:
        value += "; repeatable"
    return value


def _usage(path: str, command: click.Command) -> str:
    pieces = [path]
    if any(isinstance(param, click.Option) and not param.hidden for param in command.params):
        pieces.append("[OPTIONS]")
    for argument in (param for param in command.params if isinstance(param, click.Argument)):
        name = argument.human_readable_name.upper()
        pieces.append(name if argument.required else f"[{name}]")
    return " ".join(pieces)


def _group_for(path: str) -> str:
    for title, prefixes in GROUPS:
        if any(path == prefix or path.startswith(prefix + " ") for prefix in prefixes):
            return title
    raise KeyError(path)


def render() -> str:
    """Render a complete command reference with examples and output sketches."""

    commands = public_commands()
    if set(commands) != set(EXAMPLES) or set(commands) != set(OUTPUTS):
        missing_examples = set(commands) - set(EXAMPLES)
        missing_outputs = set(commands) - set(OUTPUTS)
        extra = (set(EXAMPLES) | set(OUTPUTS)) - set(commands)
        raise ValueError(
            "CLI documentation metadata mismatch; "
            f"missing examples={missing_examples}, missing outputs={missing_outputs}, "
            f"extra={extra}"
        )
    lines = [
        "# CLI Command Reference",
        "",
        "This page lists every public `pra` leaf command from the installed Click",
        "command tree. Hidden compatibility and research controls are intentionally",
        "excluded, exactly as they are from normal product help.",
        "",
        "Every output block is representative and abridged. Paths, versions, measured",
        "values, available devices, and recommendations depend on the local environment",
        "and supplied evidence. Use `--json` or `--yaml` where offered for automation.",
        "",
        "Use `pra COMMAND --help` as the runtime authority and this page for discoverable",
        "examples. Start with the [CLI workflow guide](cli.md) for the qualification journey.",
        "",
        "## Shared observability controls",
        "",
        "Serving, Gateway, and Agent launch commands expose the same default-off controls:",
        "`--observability`, `--otel`, `--otel-endpoint`, `--prometheus`, and",
        "`--prometheus-port`. CLI overrides take precedence over the observability file",
        "and conventional OTel environment variables. None auto-enable merely because a",
        "collector or dashboard is present. See [Observability](observability.md).",
        "",
    ]
    current_group = None
    for path, command in commands.items():
        group = _group_for(path)
        if group != current_group:
            lines.extend([f"## {group}", ""])
            current_group = group
        description = command.help or command.short_help or "Run this PRA operation."
        lines.extend(
            [
                f"### `{path}`",
                "",
                description.strip(),
                "",
                "**Usage**",
                "",
                "```text",
                _usage(path, command),
                "```",
                "",
            ]
        )
        arguments = [param for param in command.params if isinstance(param, click.Argument)]
        if arguments:
            lines.extend(
                [
                    "**Arguments**",
                    "",
                    "| Argument | Required | Description |",
                    "| --- | --- | --- |",
                ]
            )
            for argument in arguments:
                lines.append(
                    f"| `{argument.human_readable_name.upper()}` | "
                    f"{'yes' if argument.required else 'no'} | "
                    f"{ARGUMENT_HELP.get(argument.name, 'Command input value.')} |"
                )
            lines.append("")
        lines.extend(
            [
                "**Options**",
                "",
                "| Option | Value | Default | Required | Description |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for option in (
            param for param in command.params
            if isinstance(param, click.Option) and not param.hidden
        ):
            names = ", ".join(f"`{name}`" for name in (*option.opts, *option.secondary_opts))
            description_text = option.help or OPTION_HELP.get(
                option.name, f"Configure `{option.name.replace('_', '-')}`."
            )
            lines.append(
                f"| {names} | {_escape(_value(option))} | `{_default(option)}` | "
                f"{'yes' if option.required else 'no'} | {_escape(description_text)} |"
            )
        help_names = "`-h`, `--help`" if not any(
            isinstance(param, click.Option) and "-h" in param.opts for param in command.params
        ) else "`--help`"
        lines.extend(
            [
                f"| {help_names} | flag | `off` | no | Show command help and exit. |",
                "",
                "**Common use**",
                "",
                "```bash",
                EXAMPLES[path],
                "```",
                "",
                "**Example output**",
                "",
                "```text",
                OUTPUTS[path],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Exit behavior",
            "",
            "Successful commands return exit status `0`. Usage errors, unavailable optional",
            "dependencies, rejected capability requirements, and failed validation return a",
            "nonzero status. Server commands remain attached unless their command explicitly",
            "supports detaching.",
            "",
            "_Generated from `pra_hf.cli`; do not edit this page manually._",
            "",
        ]
    )
    return "\n".join(lines)


def build(*, check: bool = False) -> None:
    expected = render()
    if check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"Generated CLI reference is stale: {OUTPUT.relative_to(ROOT)}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(check=parse_args().check)
