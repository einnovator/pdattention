"""Ask a Llama-family model a question over an attached reference."""

import torch

from pra_hf import PRAConfig, PRAForCausalLM


model = PRAForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    routing_adapter="artifacts/pra_hf/routers/smollm2-135m-qasper-d128",
    pra_config=PRAConfig(selected_fraction=0.20),
    torch_dtype=torch.float16,
)
if torch.cuda.is_available():
    model.model.to("cuda")
model.add_reference("The capital of Portugal is Lisbon.")
print(model.generate("Question: What is the capital of Portugal? Answer:"))
