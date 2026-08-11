"""Minimal persistent-reference terminal chat."""

import torch

from pra_hf import PRAForCausalLM


model = PRAForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    routing_adapter="artifacts/pra_hf/routers/smollm2-135m-qasper-d128",
    torch_dtype=torch.float16,
)
if torch.cuda.is_available():
    model.model.to("cuda")
model.add_reference_file("document.txt")
messages = []
while question := input("you> ").strip():
    messages.append({"role": "user", "content": question})
    answer = model.chat(messages)
    print(f"assistant> {answer}")
    messages.append({"role": "assistant", "content": answer})
