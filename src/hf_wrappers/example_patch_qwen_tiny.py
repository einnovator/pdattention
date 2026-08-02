from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pra_wrapper import SimpleHFMemoryIndex, patch_decoder_layers

model_name = "Qwen/Qwen2.5-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    attn_implementation="eager",
)
if device == "cpu":
    model = model.to(device)

memory_texts = [
    "PRA uses URI references as latent memory handles.",
    "Each reference is resolved into layer-specific K/V caches.",
]
mem = SimpleHFMemoryIndex(memory_texts, tokenizer, model, device)
model = patch_decoder_layers(model, mem, layer_ids=[-2, -1], alpha=0.05)

inputs = tokenizer("Explain PRA briefly.", return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(out[0], skip_special_tokens=True))
