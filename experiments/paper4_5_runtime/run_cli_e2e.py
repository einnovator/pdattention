"""Run the public PRA CLI as subprocesses and emit portable evidence.

The suite uses a tiny local decoder, synthetic routing features, and a local
OpenAI-compatible endpoint. Live Hub search and bundle pull are opt-in so the
same contract can run in offline CI and in network-qualified host checks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.paper4_5_runtime.build_cli_reference import public_commands


CLI = (sys.executable, "-m", "pra_hf.cli")
HUB_BUNDLE = "EInnovator/pra-qwen3-4b-mlx-4bit"
HUB_REVISION = "49c18674ce15c8e267d5d19230d6dc152bca778b"


@dataclass
class CommandReceipt:
    command: str
    phase: str
    classification: str
    argv: list[str]
    status: str
    exit_code: int | None
    duration_seconds: float
    output_excerpt: str


class SuiteFailure(RuntimeError):
    pass


class Runner:
    def __init__(self, workspace: Path, *, live_hub: bool) -> None:
        self.workspace = workspace
        self.live_hub = live_hub
        self.receipts: list[CommandReceipt] = []
        self.covered: set[str] = set()
        self.env = os.environ.copy()
        source = str(ROOT / "src")
        self.env["PYTHONPATH"] = source + os.pathsep + self.env.get("PYTHONPATH", "")
        self.env["PRA_HOME"] = str(workspace / "pra-home")
        self.env["HF_HOME"] = str(workspace / "hf-home")
        self.env["TOKENIZERS_PARALLELISM"] = "false"
        self.env["NO_COLOR"] = "1"

    def run(
        self,
        command: str,
        arguments: Sequence[str],
        *,
        classification: str = "offline-success",
        expected_codes: Sequence[int] = (0,),
        contains: Sequence[str] = (),
        contains_any: Sequence[str] = (),
        stdin: str | None = None,
        timeout: float = 180.0,
        phase: str = "semantic",
    ) -> subprocess.CompletedProcess[str]:
        argv = [*CLI, *arguments]
        started = time.perf_counter()
        try:
            result = subprocess.run(
                argv,
                cwd=self.workspace,
                env=self.env,
                input=stdin,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or "") + (error.stderr or "")
            self._record(
                command, phase, classification, argv, "FAIL", None,
                time.perf_counter() - started, output,
            )
            raise SuiteFailure(f"{command} timed out after {timeout}s") from error
        output = result.stdout or ""
        valid = result.returncode in expected_codes
        valid = valid and all(marker in output for marker in contains)
        valid = valid and (not contains_any or any(marker in output for marker in contains_any))
        status = "PASS" if valid else "FAIL"
        self._record(
            command, phase, classification, argv, status, result.returncode,
            time.perf_counter() - started, output,
        )
        if not valid:
            raise SuiteFailure(
                f"{command} failed: exit={result.returncode}, output={output[-1200:]}"
            )
        if phase == "semantic":
            self.covered.add(command)
        return result

    def skip_external(self, command: str, arguments: Sequence[str], reason: str) -> None:
        self.receipts.append(
            CommandReceipt(
                command,
                "semantic",
                "external-prerequisite",
                [*CLI, *arguments],
                "SKIP",
                None,
                0.0,
                reason,
            )
        )
        self.covered.add(command)

    def server_receipt(
        self,
        command: str,
        argv: Sequence[str],
        started: float,
        output: str,
        status: str,
    ) -> None:
        self.receipts.append(
            CommandReceipt(
                command,
                "semantic",
                "launched-and-probed-service",
                list(argv),
                status,
                0 if status == "PASS" else None,
                time.perf_counter() - started,
                self._excerpt(output),
            )
        )
        if status == "PASS":
            self.covered.add(command)

    def _record(
        self,
        command: str,
        phase: str,
        classification: str,
        argv: Sequence[str],
        status: str,
        exit_code: int | None,
        duration: float,
        output: str,
    ) -> None:
        self.receipts.append(
            CommandReceipt(
                command,
                phase,
                classification,
                list(argv),
                status,
                exit_code,
                duration,
                self._excerpt(output),
            )
        )

    @staticmethod
    def _excerpt(output: str, limit: int = 1200) -> str:
        compact = output.strip().replace("\r\n", "\n")
        return compact[-limit:]


class StubHandler(BaseHTTPRequestHandler):
    server_version = "PRAE2E/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/pra/capabilities":
            self.send_error(404)
            return
        if self.path in {"/health", "/health_generate"}:
            self._json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "pra-e2e-stub"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path in {"/v1/chat/completions", "/v1/responses"}:
            self._json(
                200,
                {
                    "id": "pra-e2e-response",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "PRA E2E OK"}}
                    ],
                    "output": [{"content": [{"type": "output_text", "text": "PRA E2E OK"}]}],
                },
            )
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, value: Any) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _wait_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            time.sleep(0.1)
    raise SuiteFailure(f"Endpoint did not become ready: {url}: {last_error}")


def _create_tiny_model(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "PRA": 4,
        "structural": 5,
        "validation": 6,
        "Progressive": 7,
        "Retrieval": 8,
        "Attention": 9,
        "evidence": 10,
        "What": 11,
        "mechanism": 12,
        "is": 13,
        "being": 14,
        "validated": 15,
        "Name": 16,
        "the": 17,
        "attention": 18,
        ".": 19,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _write_features(path: Path) -> None:
    generator = torch.Generator().manual_seed(53)
    rows = []
    for index in range(4):
        memory = torch.randn((8, 8), generator=generator)
        query = memory[index % 8] + 0.01 * torch.randn((8,), generator=generator)
        positive = torch.zeros(8, dtype=torch.bool)
        positive[index % 8] = True
        rows.append(
            {
                "queries": {"last": query},
                "memory_gists": memory,
                "positive_mask": positive,
                "chunk_spans": [(offset * 4, offset * 4 + 4) for offset in range(8)],
                "dataset": "cli-e2e",
            }
        )
    torch.save(rows, path)


def _write_measurements(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "selector_digest": "cli-e2e-frozen-selection",
                "hardware": platform.platform(),
                "evidence_tier": "E2E_CONTROLLED",
                "modes": {
                    "full_context": {
                        "quality": {"f1": 0.8},
                        "context": {"visible_input_tokens": 1000},
                        "performance": {"ttft_p95_ms": 200.0, "successful_requests_per_second": 5.0},
                    },
                    "selected_context": {
                        "quality": {"success": True, "f1": 0.8},
                        "context": {"visible_input_tokens": 250},
                        "performance": {"ttft_p95_ms": 100.0, "successful_requests_per_second": 8.0},
                    },
                    "native_memory_hot": {
                        "quality": {"success": True, "f1": 0.8},
                        "performance": {"ttft_p95_ms": 80.0, "successful_requests_per_second": 9.0},
                        "memory": {"native_memory_bytes": 4096},
                        "lifecycle": {"reference_encoding_ms": 12.0, "reuse": 4},
                    },
                    "native_memory_warm": {
                        "quality": {"success": True, "f1": 0.8},
                        "performance": {"ttft_p95_ms": 75.0, "successful_requests_per_second": 9.2},
                        "memory": {"native_memory_bytes": 4096},
                        "lifecycle": {"reference_encoding_ms": 0.0, "reuse": 4},
                    },
                    "native_serving": {
                        "quality": {"success": True, "f1": 0.8},
                        "performance": {"ttft_p95_ms": 65.0, "successful_requests_per_second": 10.0},
                        "memory": {"native_memory_bytes": 4096},
                        "lifecycle": {"reference_encoding_ms": 0.0, "reuse": 4},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_bundle_run(path: Path, model: Path) -> None:
    structural = path / "structural_adapter"
    structural.mkdir(parents=True)
    (structural / "pra_adapter.yaml").write_text(
        "schema_version: 1\nsource: cli-e2e\n", encoding="utf-8"
    )
    runtime = {
        "schema_version": 1,
        "model": {
            "id": str(model),
            "revision": "cli-e2e-local",
            "architecture": "LlamaForCausalLM",
        },
        "structural_adapter": {"path": "structural_adapter", "status": "validated"},
        "learned_adapters": {},
        "default_profile": "REFERENCE_CORRECTNESS",
        "profiles": {"REFERENCE_CORRECTNESS": {"status": "CONTROLLED"}},
        "qualification": {"status": "CONTROLLED", "metrics": []},
        "runtime_compatibility": {"hf": "controlled"},
        "provenance": {"license": "apache-2.0", "pra_commit": _git_sha()},
        "trust": {"status": "local/private", "publisher": "cli-e2e"},
    }
    (path / "pra.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "NOT_MEASURED"


def _run_gateway(runner: Runner, stub_url: str) -> None:
    command = "pra gateway serve"
    port = _free_port()
    argv = [
        *CLI,
        "gateway",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mode",
        "selected-context",
        "--backend",
        "openai",
        "--backend-url",
        stub_url,
    ]
    log_path = runner.workspace / "gateway.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            cwd=runner.workspace,
            env=runner.env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    status = "FAIL"
    try:
        health = _wait_json(f"http://127.0.0.1:{port}/health")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "pra-e2e-stub",
                    "messages": [{"role": "user", "content": "Return the E2E marker."}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            value = json.loads(response.read().decode("utf-8"))
        if health.get("status") == "ok" and value["choices"][0]["message"]["content"] == "PRA E2E OK":
            status = "PASS"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        output = log_path.read_text(encoding="utf-8", errors="replace")
        runner.server_receipt(command, argv, started, output, status)
    if status != "PASS":
        raise SuiteFailure("Gateway did not proxy the controlled OpenAI request.")


def _run_managed_hf_serve(
    runner: Runner,
    command: str,
    arguments: Sequence[str],
) -> None:
    """Exercise managed serving and tear down the exact process it launches."""

    result = runner.run(
        command,
        arguments,
        classification="launched-and-probed-service",
        contains=('"engine": "hf"', '"managed": true'),
        timeout=120,
    )
    try:
        process_id = int(json.loads(result.stdout)["process_id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SuiteFailure(f"{command} did not return a managed process id") from error
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process_id), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.kill(process_id, signal.SIGTERM)
    except ProcessLookupError:
        # A startup failure may exit before cleanup; the CLI contract was still observed.
        pass


def _run_help_contract(runner: Runner, commands: Sequence[str]) -> None:
    for command in commands:
        arguments = [*command.split()[1:], "--help"]
        runner.run(
            command,
            arguments,
            classification="help-contract",
            contains=("Usage:",),
            phase="help",
            timeout=45,
        )


def _run_semantics(runner: Runner, stub_url: str) -> None:
    root = runner.workspace
    model = root / "tiny-llama"
    adapter = root / "structural-adapter"
    onboard = root / "onboard"
    features = root / "features.pt"
    router = root / "router"
    measurements = root / "measurements.json"
    run = root / "qualification"
    native_run = root / "native-qualification"
    serving_run = root / "serving-qualification"
    assessment_root = root / "assessments"
    assessment = assessment_root / "customer"
    profile_run = root / "profile-run"
    bundle_run = root / "bundle-run"
    bundle = root / "bundle"
    runtime = root / "runtime"
    benchmark = root / "benchmark"
    report = root / "report.html"

    _create_tiny_model(model)
    _write_features(features)
    _write_measurements(measurements)
    _write_bundle_run(bundle_run, model)

    runner.run("pra doctor", ["doctor", "--json"], contains=('"system"',))
    runner.run("pra engines", ["engines", "--json"], contains=('"engines"',))
    runner.run("pra inspect", ["inspect", str(model), "-e", "hf", "-a", "none", "--json"], contains=('"requested_engine": "hf"',))
    runner.run("pra evaluate", ["evaluate", str(model), "-e", "mlx", "-D", "cli-e2e", "--measurements", str(measurements), "-a", "none", "-o", str(run), "--json"], contains=('"digest": "cli-e2e-frozen-selection"',))
    runner.run("pra recommend", ["recommend", str(run), "--json"], contains=('"recommended_mode"',))
    runner.run("pra report", ["report", str(run), "--format", "html", "-o", str(report)], contains=(str(report),))
    runner.run("pra qualify native-memory", ["qualify", "native-memory", str(model), "-e", "mlx", "-D", "cli-e2e", "--measurements", str(measurements), "-o", str(native_run), "--json"], contains=('"native_memory_hot"',))
    runner.run("pra qualify native-serving", ["qualify", "native-serving", str(model), "-e", "vllm", "-D", "cli-e2e", "--measurements", str(measurements), "-o", str(serving_run), "--json"], contains=('"native_serving"',))

    runner.run("pra assess init", ["assess", "init", "customer", "--root", str(assessment_root)], contains=(str(assessment),))
    (assessment / "config.yaml").write_text(
        yaml.safe_dump({"schema_version": "1.0", "name": "customer", "model": str(model), "engine": "mlx", "dataset": "cli-e2e", "profile": "recommended"}),
        encoding="utf-8",
    )
    runner.run("pra assess run", ["assess", "run", str(assessment), "--measurements", str(measurements), "--json"], contains=('"digest": "cli-e2e-frozen-selection"',))
    runner.run("pra assess report", ["assess", "report", str(assessment), "--format", "html"], contains=("report.html",))

    runner.run("pra model inspect", ["model", "inspect", str(model), "--json"], contains=("LlamaForCausalLM",))
    runner.run("pra model adapt", ["model", "adapt", str(model), "-o", str(adapter), "--json"], contains=("pra_adapter.yaml",))
    runner.run("pra model validate", ["model", "validate", str(model), "-a", str(adapter), "-d", "cpu", "-o", str(root / "model-validation"), "--json"], contains=('"V1_model_load"',), timeout=240)
    runner.run("pra model onboard", ["model", "onboard", str(model), "-o", str(onboard), "--json"], contains=('"runs"',))

    runner.run("pra adapter train routing", ["adapter", "train", "routing", str(model), "--model-family", "llama", "--train-features", str(features), "--validation-features", str(features), "--steps", "2", "--routing-dim", "8", "-d", "cpu", "-o", str(router), "--json"], contains=('"steps": 2',), timeout=180)
    runner.run("pra adapter inspect", ["adapter", "inspect", str(router), "--json"], contains=('"routing_dim": 8',))
    runner.run("pra adapter eval", ["adapter", "eval", str(router), "--features", str(features), "-d", "cpu", "--json"], contains=('"query_strategy": "last"',))
    runner.run("pra adapter train memory", ["adapter", "train", "memory"], classification="expected-policy-rejection", expected_codes=(1,), contains=("research-only",))
    runner.run("pra adapter train late-band", ["adapter", "train", "late-band"], classification="expected-policy-rejection", expected_codes=(1,), contains=("research-only",))

    runner.run("pra profiles show", ["profiles", "show", str(model), "--json"], contains=('"profiles"',))
    runner.run("pra profiles calibrate", ["profiles", "calibrate", str(model), "-o", str(profile_run), "--json"], contains=("CALIBRATION_PENDING",))
    runner.run("pra profiles compare", ["profiles", "compare", str(model), "--json"], contains=('"profiles"',))
    runner.run("pra profiles report", ["profiles", "report", str(model), "-o", str(root / "profiles.md")], contains=("profiles.md",))

    runner.run("pra bundle build", ["bundle", "build", str(bundle_run), "-o", str(bundle), "--json"], contains=('"schema_version": 2',))
    runner.run("pra bundle inspect", ["bundle", "inspect", str(bundle), "--json"], contains=('"schema_version": 2',))
    runner.run("pra bundle validate", ["bundle", "validate", str(bundle), "--json"], contains=('"status": "VALID"',))
    runner.run("pra bundle card", ["bundle", "card", str(bundle), "--update"], contains=("README.md",))
    runner.run("pra bundle list", ["bundle", "list", "--family", "qwen", "--json"], contains=('"count": 3',))
    runner.run("pra bundle resolve", ["bundle", "resolve", str(model), "-e", "hf", "-a", "none", "--json"], contains=('"status": "DISABLED"',))

    runner.run("pra hf login", ["hf", "login", "--check", "--json"], classification="authenticated-or-guarded", expected_codes=(0, 1), contains_any=("AUTHENTICATED", "No usable Hugging Face authentication"))
    runner.run("pra hf list", ["hf", "list", "--family", "qwen", "--json"], contains=('"source": "trusted-registry"',))
    if runner.live_hub:
        runner.run("pra hf search", ["hf", "search", "qwen", "--author", "EInnovator", "--limit", "5", "--json"], classification="live-hub", contains=(HUB_BUNDLE,), timeout=120)
        pulled = root / "pulled-bundle"
        runner.run("pra hf pull", ["hf", "pull", HUB_BUNDLE, "-r", HUB_REVISION, "-o", str(pulled), "--json"], classification="live-hub", contains=(HUB_REVISION,), timeout=180)
        runner.run("pra hf inspect", ["hf", "inspect", str(pulled), "--json"], classification="live-hub", contains=('"schema_version": 2',))
    else:
        runner.skip_external("pra hf search", ["hf", "search", "qwen"], "Live Hub access disabled for this run.")
        runner.skip_external("pra hf pull", ["hf", "pull", HUB_BUNDLE], "Live Hub access disabled for this run.")
        runner.run("pra hf inspect", ["hf", "inspect", str(bundle), "--json"], contains=('"schema_version": 2',))
    runner.run("pra hf push", ["hf", "push", str(bundle), "example/pra-cli-e2e", "--dry-run", "--json"], contains=('"dry_run": true',))
    manifest = root / "publish.yaml"
    manifest.write_text(yaml.safe_dump({"bundles": [{"bundle": str(bundle), "repo_id": "example/pra-cli-e2e"}]}), encoding="utf-8")
    runner.run("pra hf publish-manifest", ["hf", "publish-manifest", str(manifest), "--dry-run", "--json"], contains=('"status": "VALIDATED"',))

    runner.run("pra runtime init", ["runtime", "init", str(runtime), "--max-native-index-tokens", "1024", "--defer-native-index"], contains=("pra_runtime_config.json",))
    _run_managed_hf_serve(
        runner,
        "pra runtime serve",
        ["runtime", "serve", str(model), "-m", "selected-context", "-e", "hf", "-d", "cpu", "-a", "none", "--port", str(_free_port()), "--json"],
    )
    runner.run("pra runtime inspect", ["runtime", "inspect", "pra-e2e-stub", "-e", "openai", "-u", stub_url, "-a", "none", "--json"], contains=('"engine": "openai"',))
    runner.run("pra runtime doctor", ["runtime", "doctor", "-e", "openai", "-u", stub_url, "--json"], contains=('"status": "AVAILABLE"',))
    runner.run("pra runtime benchmark", ["runtime", "benchmark", "pra-e2e-stub", "-e", "openai", "-d", "cpu", "-o", str(benchmark), "--json"], contains=('"artifacts"',), timeout=300)
    runner.run("pra runtime capabilities", ["runtime", "capabilities", "--json"], contains=('"platform"', '"torch_compile_api"'))
    _run_managed_hf_serve(
        runner,
        "pra serve",
        ["serve", str(model), "-m", "selected-context", "-e", "hf", "-d", "cpu", "-a", "none", "--port", str(_free_port()), "--json"],
    )

    _run_gateway(runner, stub_url)
    agent_config = root / "agent.yaml"
    web_port = _free_port()
    agent_config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "default_profile": "e2e",
                "profiles": {
                    "e2e": {
                        "model": "pra-e2e-stub",
                        "runtime": {"mode": "remote", "engine": "openai", "endpoint": stub_url},
                        "workspace": str(root),
                        "sessions": {"path": str(root / "sessions")},
                        "tools": {"approval": "deny", "max_rounds": 0},
                        "context": {"transport": "text", "allow_text_fallback": True},
                        "generation": {"max_new_tokens": 8},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner.run("pra agent inspect", ["agent", "inspect", "-c", str(agent_config), "--json"], contains=('"name": "e2e"',))
    runner.run("pra agent run", ["agent", "run", "Return the marker", "-c", str(agent_config), "--json"], contains=("PRA E2E OK",))
    runner.run("pra agent chat", ["agent", "chat", "-c", str(agent_config)], stdin="/status\n/exit\n", contains=("PRA Agent", "session="))
    try:
        runner.run(
            "pra agent start",
            ["agent", "start", "-c", str(agent_config), "--host", "127.0.0.1", "--port", str(web_port), "--detach", "--json"],
            classification="launched-and-probed-service",
            contains=('"pid"', '"url"'),
            timeout=60,
        )
    finally:
        runner.run("pra agent stop", ["agent", "stop"], contains=("STOPPED",), timeout=60)


def run(
    output: Path, *, live_hub: bool, include_help_contracts: bool = True
) -> dict[str, Any]:
    commands = tuple(public_commands())
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="pra-cli-e2e-") as temporary:
        workspace = Path(temporary)
        runner = Runner(workspace, live_hub=live_hub)
        stub = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        thread = threading.Thread(target=stub.serve_forever, daemon=True)
        thread.start()
        try:
            if include_help_contracts:
                _run_help_contract(runner, commands)
            _run_semantics(runner, f"http://127.0.0.1:{stub.server_port}")
        finally:
            stub.shutdown()
            stub.server_close()
            thread.join(timeout=5)

        missing = sorted(set(commands) - runner.covered)
        extra = sorted(runner.covered - set(commands))
        failures = [row for row in runner.receipts if row.status == "FAIL"]
        if missing or extra or failures:
            raise SuiteFailure(
                f"CLI coverage mismatch: missing={missing}, extra={extra}, failures={len(failures)}"
            )
        report = {
            "schema_version": "pra-cli-e2e-v1",
            "status": "PASS",
            "git_commit": _git_sha(),
            "started_at_unix": started,
            "duration_seconds": time.time() - started,
            "host": {
                "node": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            },
            "live_hub": live_hub,
            "public_leaf_commands": len(commands),
            "help_contracts_requested": include_help_contracts,
            "help_contracts_passed": sum(row.phase == "help" and row.status == "PASS" for row in runner.receipts),
            "semantic_commands_covered": len(runner.covered),
            "semantic_passed": sum(row.phase == "semantic" and row.status == "PASS" for row in runner.receipts),
            "semantic_skipped": sum(row.phase == "semantic" and row.status == "SKIP" for row in runner.receipts),
            "receipts": [asdict(row) for row in runner.receipts],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-hub", action="store_true")
    parser.add_argument("--semantic-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        args.output,
        live_hub=args.live_hub,
        include_help_contracts=not args.semantic_only,
    )
    print(json.dumps({key: report[key] for key in (
        "status", "git_commit", "duration_seconds", "host", "live_hub",
        "public_leaf_commands", "help_contracts_passed",
        "semantic_commands_covered", "semantic_passed", "semantic_skipped",
    )}, indent=2))


if __name__ == "__main__":
    main()
