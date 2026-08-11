"""Ask Qwen a question over a CPU-resident text reference."""

import torch

from pra_hf import PRAConfig, PRAForCausalLM


model = PRAForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    routing_adapter="artifacts/pra_hf/routers/qwen3-0.6b-qasper-d128",
    pra_config=PRAConfig(selected_fraction=0.20),
    torch_dtype=torch.float16,
)
if torch.cuda.is_available():
    model.model.to("cuda")
model.add_reference_file("document.txt")
result = model.generate("Question: What does the document conclude?", return_details=True)
print(result.text)
print(result.stats["materialized_kv_token_fraction"])
