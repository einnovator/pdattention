import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data.collators import PRACollator
from data.datamodules import PRADataModule
from .cache_services import build_cache_from_metadata
from .config import PRAConfig
from .data import CharTokenizer, build_training_corpus
from data.tokenizer import BPETokenizer
from .model import TinyPRAModel


@dataclass
class EvalResult:
    name: str
    exact: int
    total: int
    avg_input_tokens: float
    avg_resolved_refs: float
    answer_exact_match: float
    expected_ref_hit: float
    expected_anchor_hit: float
    avg_num_expansions: float
    avg_retrieved_tokens: float
    avg_full_context_tokens: float
    lm_loss: float
    cache_hit_ratio: float
    latency: float
    sample_output: str

    @property
    def accuracy(self) -> float:
        return self.exact / max(self.total, 1)


def load_model_and_tokenizer(
    ckpt_path: str,
    device: str,
    examples,
    *,
    max_seq_len: int = 96,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    n_vanilla_layers: int = 0,
    n_mixed_layers: int = 0,
    dropout: float = 0.0,
    pra_layer_ids: tuple[int, ...] = (2, 3),
    top_k_references: int = 2,
    top_k_chunks_per_reference: int = 1,
    trigger_threshold: float = 0.2,
    use_cross_attention_memory: bool = True,
    use_concat_memory: bool = False,
    memory_alpha: float = 0.5,
    resolver_config=None,
    cache_config=None,
    **routing_kwargs,
):
    if not os.path.exists(ckpt_path):
        corpus = build_training_corpus(examples)
        tokenizer = CharTokenizer(corpus)
        cfg = PRAConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_vanilla_layers=n_vanilla_layers,
            n_mixed_layers=n_mixed_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pra_layer_ids=tuple(pra_layer_ids),
            top_k_references=top_k_references,
            top_k_chunks_per_reference=top_k_chunks_per_reference,
            trigger_threshold=trigger_threshold,
            use_cross_attention_memory=use_cross_attention_memory,
            use_concat_memory=use_concat_memory,
            memory_alpha=memory_alpha,
            device=device,
            **routing_kwargs,
        )
        model = TinyPRAModel(cfg).to(device)
        return model, tokenizer, False

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg_dict = dict(checkpoint["cfg"])
    cfg_dict["device"] = device
    cfg = PRAConfig(**cfg_dict)
    tokenizer = (
        BPETokenizer.from_json(checkpoint["tokenizer_json"])
        if checkpoint.get("tokenizer_type") == "BPETokenizer" and checkpoint.get("tokenizer_json")
        else CharTokenizer.from_vocab(checkpoint["stoi"])
    )
    model = TinyPRAModel(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, True


def generate_greedy(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx = input_ids[:, -model.cfg.max_seq_len :]
        logits = model(idx, use_pra_memory=False)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return tokenizer.decode(input_ids[0])


def generate_pra(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
        do_sample=False,
    )
    return tokenizer.decode(generated[0])


def baseline_prompts_from_metadata(item):
    refs = item["references"]
    full_docs = "\n".join(str(ref.metadata.get("text", "")) for ref in refs)
    summaries = "\n".join(ref.summary or "" for ref in refs)
    question = item["question"]
    return {
        "no_refs": question,
        "full_context": f"Context:\n{full_docs}\n\n{question}",
        "simple_rag": f"Retrieved summary:\n{summaries}\n\n{question}",
        "pra": question,
    }


def baseline_prompts(ex):
    if hasattr(ex, "references"):
        return baseline_prompts_from_metadata({"question": ex.question, "references": ex.references})
    full_docs = "\n".join(ex.refs.values())
    summaries = "\n".join(ex.summaries.values())
    return {
        "no_refs": ex.prompt,
        "full_context": f"Context:\n{full_docs}\n\n{ex.prompt}",
        "simple_rag": f"Retrieved summary:\n{summaries}\n\n{ex.prompt}",
        "pra": ex.prompt,
    }


def generated_contains_target(output: str, prompt: str, target: str) -> bool:
    return target.strip() in output[len(prompt) :]


def evaluate_variant(name, model, tokenizer, dataloader, device, max_new_tokens, *, resolver_config=None, cache_config=None):
    start = time.perf_counter()
    exact = 0
    ref_hits = 0
    anchor_hits = 0
    cache_hits = 0
    token_cost = 0
    resolved_refs = 0
    num_expansions = 0
    retrieved_tokens = 0
    full_context_tokens = 0
    loss_sum = 0.0
    loss_count = 0
    sample_output = ""
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if name == "pra":
            cache = build_cache_from_metadata(
                model,
                tokenizer,
                batch["metadata"],
                device,
                resolver_config=resolver_config,
                cache_config=cache_config,
            )
            resolved_refs += len(cache.entries)
            retrieved_tokens += sum(len(tokenizer.encode(entry.text)) for entry in cache.all_entries())
            logits = model(input_ids)
        else:
            model.clear_pra_cache()
            logits = model(input_ids, use_pra_memory=False)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        loss_sum += float(loss.detach().cpu())
        loss_count += 1

        for i, item in enumerate(batch["metadata"]):
            prompts = baseline_prompts_from_metadata(item)
            prompt = prompts[name]
            target = item["answer"].strip()
            token_cost += len(tokenizer.encode(prompt))
            full_context_tokens += len(tokenizer.encode(prompts["full_context"]))

            if name == "pra":
                expected_uris = {ref.uri for ref in item["references"] if ref.id in item["target_reference_ids"]}
                if not expected_uris or expected_uris.intersection(model.pra_cache.entries):
                    ref_hits += 1
                    cache_hits += 1
                expected_anchors = item.get("expected_anchors", [])
                if not expected_anchors:
                    anchor_hits += 1
                else:
                    cached_text = "\n".join(entry.text for entry in model.pra_cache.all_entries())
                    if any(anchor in cached_text or anchor in " ".join(model.pra_cache.entries) for anchor in expected_anchors):
                        anchor_hits += 1
                output = generate_pra(model, tokenizer, prompt, device, max_new_tokens)
            else:
                output = generate_greedy(model, tokenizer, prompt, device, max_new_tokens)

            if generated_contains_target(output, prompt, target):
                exact += 1
            if total == 0:
                sample_output = output
            total += 1

    latency = time.perf_counter() - start
    return EvalResult(
        name=name,
        exact=exact,
        total=total,
        avg_input_tokens=token_cost / max(total, 1),
        avg_resolved_refs=resolved_refs / max(total, 1),
        answer_exact_match=exact / max(total, 1),
        expected_ref_hit=ref_hits / max(total, 1) if name == "pra" else 0.0,
        expected_anchor_hit=anchor_hits / max(total, 1) if name == "pra" else 0.0,
        avg_num_expansions=num_expansions / max(total, 1),
        avg_retrieved_tokens=retrieved_tokens / max(total, 1),
        avg_full_context_tokens=full_context_tokens / max(total, 1),
        lm_loss=loss_sum / max(loss_count, 1),
        cache_hit_ratio=cache_hits / max(total, 1) if name == "pra" else 0.0,
        latency=latency,
        sample_output=sample_output,
    )


def print_report(results, loaded_checkpoint: bool):
    ckpt_note = "checkpoint" if loaded_checkpoint else "random initialization"
    print(f"PRA eval report ({ckpt_note})")
    print("variant       loss     accuracy   input_tokens   resolved_refs   ref_hit   anchor_hit   cache_hit   latency")
    for result in results:
        print(
            f"{result.name:<13} "
            f"{result.lm_loss:>6.3f} "
            f"{result.accuracy:>7.2%} "
            f"{result.avg_input_tokens:>14.1f} "
            f"{result.avg_resolved_refs:>15.1f} "
            f"{result.expected_ref_hit:>8.2%} "
            f"{result.expected_anchor_hit:>11.2%} "
            f"{result.cache_hit_ratio:>10.2%} "
            f"{result.latency:>8.3f}s"
        )
    print("\nSample PRA output:")
    pra_result = next((r for r in results if r.name == "pra"), None)
    if pra_result is not None:
        print(pra_result.sample_output)


def run_evaluation(
    *,
    ckpt: str,
    device: str,
    datamodule,
    max_new_tokens: int,
    max_seq_len: int = 96,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    n_vanilla_layers: int = 0,
    n_mixed_layers: int = 0,
    dropout: float = 0.0,
    pra_layer_ids: tuple[int, ...] = (2, 3),
    top_k_references: int = 2,
    top_k_chunks_per_reference: int = 1,
    trigger_threshold: float = 0.2,
    use_cross_attention_memory: bool = True,
    use_concat_memory: bool = False,
    memory_alpha: float = 0.5,
    resolver_config=None,
    cache_config=None,
    **routing_kwargs,
):
    model, tokenizer, loaded_checkpoint = load_model_and_tokenizer(
        ckpt,
        device,
        datamodule.dataset,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_vanilla_layers=n_vanilla_layers,
        n_mixed_layers=n_mixed_layers,
        dropout=dropout,
        pra_layer_ids=pra_layer_ids,
        top_k_references=top_k_references,
        top_k_chunks_per_reference=top_k_chunks_per_reference,
        trigger_threshold=trigger_threshold,
        use_cross_attention_memory=use_cross_attention_memory,
        use_concat_memory=use_concat_memory,
        memory_alpha=memory_alpha,
        **routing_kwargs,
    )
    datamodule.tokenizer = tokenizer
    datamodule.collator = PRACollator(tokenizer, max_seq_len=max_seq_len)
    variants = ["no_refs", "full_context", "simple_rag", "pra"]
    results = [
        evaluate_variant(
            name,
            model,
            tokenizer,
            datamodule.test_loader(),
            device,
            max_new_tokens,
            resolver_config=resolver_config,
            cache_config=cache_config,
        )
        for name in variants
    ]
    print_report(results, loaded_checkpoint)
    return results


def main():
    from .cli import eval as eval_command

    eval_command.main(prog_name="python -m pra_torch.eval")


if __name__ == "__main__":
    main()
