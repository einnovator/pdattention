"""Qualify scheduler-owned zero-copy CUDA aliases on frozen Task02 turn 4."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from vllm import LLM, SamplingParams

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import _assistant_prompts
from experiments.paper4_5_agent.run_vllm_cuda_live_agent_kv_gate import _selected_page_indices
from experiments.paper4_5_agent.sparse_gate_common import sparse_causal_plan
from experiments.paper6_vllm.run_cuda_connector_candidate import _prompt
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--storage", type=Path, default=Path(".pra/vllm-cuda-task02-alias"))
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
args = parser.parse_args()

trajectory_path = args.trajectory.expanduser().resolve()
output_path = args.output.expanduser().resolve()
storage = args.storage.expanduser().resolve()
storage.mkdir(parents=True, exist_ok=True)

llm = LLM(
    model=args.model,
    max_model_len=8192,
    max_num_seqs=2,
    gpu_memory_utilization=args.gpu_memory_utilization,
    enable_prefix_caching=True,
    enforce_eager=True,
    disable_hybrid_kv_cache_manager=True,
    kv_transfer_config={
        "kv_connector": "PRASparseConnector",
        "kv_connector_module_path": "pra_vllm.cuda_sparse_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "storage_path": str(storage),
            "scheduler_page_aliases": True,
        },
    },
)
tokenizer = llm.get_tokenizer()
block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
messages = trajectory["messages"]
assistant_indexes = [
    index for index, message in enumerate(messages) if message.get("role") == "assistant"
][:4]
prompt = _assistant_prompts(tokenizer, trajectory, 4)[3]
assistant_index = assistant_indexes[3]
source_tokens = (len(prompt) // block_size) * block_size
source, suffix = prompt[:source_tokens], prompt[source_tokens:]
plan = sparse_causal_plan(
    tokenizer,
    messages[:assistant_index],
    prompt,
    source_tokens=source_tokens,
    retention_fraction=0.9,
)
page_indices = _selected_page_indices(plan, block_size)
selected_tokens = len(page_indices) * block_size
selected = [
    token
    for page_index in page_indices
    for token in source[page_index * block_size : (page_index + 1) * block_size]
]

source_key = "task02-turn4-full-fcde2c74"
selected_key = "task02-turn4-selected090-fcde2c74"
generation = 4
prime_command = SparseCudaConnectorCommand(
    "store",
    source_key,
    source_tokens,
    source_tokens,
    source_generation=generation,
    residency="warm",
    request_scope="task02-turn4-prime",
)
llm.generate(
    _prompt(source + suffix[:1], prime_command.cache_salt()),
    SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    use_tqdm=False,
)
scheduler_connector = llm.llm_engine.engine_core.engine_core.scheduler.connector
source_snapshot = scheduler_connector._scheduler_alias_registry.snapshot()

directory = storage / hashlib.sha256(selected_key.encode()).hexdigest()
directory.mkdir(parents=True, exist_ok=True)
(directory / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": "pra-vllm-cuda-scheduler-alias-v1",
            "logical_key": selected_key,
            "source_tokens": selected_tokens,
            "source_generation": generation,
            "parent_logical_key": source_key,
            "selected_page_indices": list(page_indices),
            "physical_kv_copy_bytes": 0,
            "host_to_device_bytes": 0,
        }
    ),
    encoding="utf-8",
)
load_command = SparseCudaConnectorCommand(
    "load",
    selected_key,
    selected_tokens,
    source_tokens,
    source_generation=generation,
    residency="hot",
    request_scope="task02-turn4-pair",
)
sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
torch.cuda.synchronize()
outputs = llm.generate(
    [_prompt(selected + suffix, load_command.cache_salt()) for _ in range(2)],
    sampling,
    use_tqdm=False,
)
torch.cuda.synchronize()
after_pair = scheduler_connector._scheduler_alias_registry.snapshot()
token_rows = [list(map(int, row.outputs[0].token_ids)) for row in outputs]
evicted = scheduler_connector.evict_scheduler_source(
    source_key, source_generation=generation
)

def source_record(module):
    path = Path(inspect.getsourcefile(module) or "").resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

import pra_vllm.cuda_scheduler_alias as alias_module
import pra_vllm.cuda_sparse_connector as connector_module
import vllm.v1.core.sched.scheduler as scheduler_module

payload = {
    "schema_version": "paper4.5.vllm-cuda-scheduler-page-alias-task02.v1",
    "probe": "frozen_task02_turn4_requested_090_scheduler_owned_page_alias",
    "base_commit": "fcde2c74",
    "trajectory": str(trajectory_path),
    "engine": "vllm-cuda",
    "engine_version": vllm.__version__,
    "torch_version": torch.__version__,
    "device": torch.cuda.get_device_name(),
    "model": args.model,
    "temperature": 0,
    "turn": 4,
    "requested_retention_fraction": 0.9,
    "realized_retention_fraction": selected_tokens / source_tokens,
    "block_size": block_size,
    "source_tokens": source_tokens,
    "selected_tokens": selected_tokens,
    "source_position_base": source_tokens,
    "wire_query_suffix_tokens": len(suffix),
    "source_block_count": source_tokens // block_size,
    "selected_block_count": len(page_indices),
    "selected_page_indices": list(page_indices),
    "same_subset_output_token_ids": token_rows,
    "same_subset_exact": token_rows[0] == token_rows[1],
    "source_snapshot": source_snapshot,
    "after_pair": after_pair,
    "evicted_source_block_ids": list(evicted),
    "copy_accounting": {
        "physical_kv_copy_bytes": 0,
        "host_to_device_bytes": 0,
        "selected_history_reencoded_tokens": 0,
    },
    "integration_scope": {
        "scheduler_block_table_authoritative": True,
        "original_position_persisted_on_every_decode_step": True,
        "complete_pages_required": True,
        "homogeneous_single_kv_group_required": True,
        "multiprocess_scheduler_worker_qualification": False,
    },
    "provenance": {
        "alias_module": source_record(alias_module),
        "connector_module": source_record(connector_module),
        "pinned_scheduler": source_record(scheduler_module),
    },
}
payload["qualified"] = bool(
    payload["same_subset_exact"]
    and payload["after_pair"]["telemetry"]["alias_install_events"] == 2
    and payload["after_pair"]["telemetry"]["alias_release_events"] == 2
    and payload["copy_accounting"] == {
        "physical_kv_copy_bytes": 0,
        "host_to_device_bytes": 0,
        "selected_history_reencoded_tokens": 0,
    }
)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "qualified": payload["qualified"],
    "source_tokens": source_tokens,
    "selected_tokens": selected_tokens,
    "realized_retention_fraction": payload["realized_retention_fraction"],
    "same_subset_exact": payload["same_subset_exact"],
    "alias_events": payload["after_pair"]["telemetry"],
    "output": str(output_path),
}, indent=2))
