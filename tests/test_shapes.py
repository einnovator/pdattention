import torch
from pra_torch.config import PRAConfig
from pra_torch.model import PRASATransformerBlock, PRATransformerBlock, TinyPRALanguageModel, VanillaTransformerBlock
from pra_torch.data import CharTokenizer
from pra_torch.resolver import InMemoryResolver
from pra_torch.memory import PRASimpleMemoryCache


def test_model_forward_shape():
    tok = CharTokenizer(["hello world"])
    cfg = PRAConfig(vocab_size=tok.vocab_size, max_seq_len=16, d_model=32, n_heads=4, n_layers=2)
    model = TinyPRALanguageModel(cfg)
    x = torch.randint(0, tok.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, tok.vocab_size)


def test_reference_cache_shapes():
    tok = CharTokenizer(["abc", "summary"])
    cfg = PRAConfig(vocab_size=tok.vocab_size, max_seq_len=16, d_model=32, n_heads=4, n_layers=2)
    model = TinyPRALanguageModel(cfg)
    entry = model.encode_reference_to_cache("mem://x", "abc", "summary", tok, "cpu")
    assert 0 in entry.layer_kv
    assert entry.layer_kv[0].k.ndim == 4
    assert entry.layer_kv[0].v.ndim == 4


def test_model_forward_with_cache():
    tok = CharTokenizer(["question <REF_1> answer", "secret code red"])
    cfg = PRAConfig(vocab_size=tok.vocab_size, max_seq_len=24, d_model=32, n_heads=4, n_layers=2, trigger_threshold=-1.0)
    model = TinyPRALanguageModel(cfg)
    entry = model.encode_reference_to_cache("mem://x", "secret code red", "secret code", tok, "cpu")
    cache = PRASimpleMemoryCache()
    cache.put(entry)
    model.set_pra_cache(cache)
    x = torch.tensor([tok.encode("question <REF_1> answer")[:24]])
    logits = model(x)
    assert logits.shape[0] == 1


def test_model_accepts_memory_cache_at_construction():
    tok = CharTokenizer(["question <REF_1> answer", "secret code red"])
    cfg = PRAConfig(vocab_size=tok.vocab_size, max_seq_len=24, d_model=32, n_heads=4, n_layers=2)
    cache = PRASimpleMemoryCache()
    model = TinyPRALanguageModel(cfg, pra_cache=cache)
    assert model.pra_cache is cache
    assert all(block.attn.pra_cache is cache for block in model.blocks)


def test_model_builds_ordered_vanilla_mixed_and_pra_layers():
    tok = CharTokenizer(["question <REF_1> answer", "secret code red"])
    cfg = PRAConfig(
        vocab_size=tok.vocab_size,
        max_seq_len=24,
        d_model=32,
        n_heads=4,
        n_layers=4,
        n_vanilla_layers=1,
        n_mixed_layers=1,
        trigger_threshold=-1.0,
    )
    model = TinyPRALanguageModel(cfg)
    assert [type(block) for block in model.blocks] == [
        VanillaTransformerBlock,
        PRASATransformerBlock,
        PRATransformerBlock,
        PRATransformerBlock,
    ]

    entry = model.encode_reference_to_cache("mem://x", "secret code red", "secret code", tok, "cpu")
    assert sorted(entry.layer_kv) == [1, 2, 3]


def test_explicit_model_variants_select_expected_blocks():
    sa = TinyPRALanguageModel(
        PRAConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=4, model_variant="td_sa")
    )
    pra = TinyPRALanguageModel(
        PRAConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=4, model_variant="td_pra")
    )
    mixed = TinyPRALanguageModel(
        PRAConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=4, model_variant="tdx_pra")
    )

    assert all(isinstance(block, VanillaTransformerBlock) for block in sa.blocks)
    assert all(isinstance(block, PRATransformerBlock) for block in pra.blocks)
    assert [type(block) for block in mixed.blocks] == [
        VanillaTransformerBlock,
        VanillaTransformerBlock,
        PRATransformerBlock,
        PRATransformerBlock,
    ]
