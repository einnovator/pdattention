from pra_core.datasets import load_dataset
from pra_torch.data import CharTokenizer
from pra_torch.eval import baseline_prompts, generated_contains_target


def test_tokenizer_from_vocab_round_trips():
    tok = CharTokenizer(["abc <REF_1>"])
    restored = CharTokenizer.from_vocab(tok.stoi)
    text = "cab <REF_1>"
    assert restored.decode(restored.encode(text)) == text


def test_ref_tokens_are_atomic():
    tok = CharTokenizer(["abc <REF_12>"])
    assert tok.encode("<REF_12>") == [tok.stoi["<REF_12>"]]


def test_baseline_prompts_include_expected_variants():
    ex = load_dataset("stage0_synthetic_memory", "data", max_examples=1)[0]
    prompts = baseline_prompts(ex)
    assert set(prompts) == {"no_refs", "full_context", "simple_rag", "pra"}
    assert "Context:" in prompts["full_context"]
    assert "Retrieved summary:" in prompts["simple_rag"]
    assert prompts["pra"] == prompts["no_refs"]


def test_generated_contains_target_ignores_prompt_context():
    prompt = "Context has answer red-lynx-1. Answer:"
    assert not generated_contains_target(prompt + " nope", prompt, "red-lynx-1")
    assert generated_contains_target(prompt + " red-lynx-1", prompt, "red-lynx-1")
