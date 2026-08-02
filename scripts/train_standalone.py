from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.datamodules import PRADataModule
from pra_torch.cli import load_config
from pra_torch.config import CacheServiceConfig, PRAConfig, ResolverServiceConfig, TrainConfig
from pra_torch.trainer import PRAStandaloneTrainer


def train_config_from_dict(raw: dict) -> TrainConfig:
    standalone = raw.get("standalone", raw)
    keys = {field.name for field in fields(TrainConfig)}
    values = {k: v for k, v in standalone.items() if k in keys}
    values.setdefault("resolver_config", raw.get("resolver"))
    values.setdefault("cache_config", raw.get("cache"))
    values["resolver_config"] = ResolverServiceConfig.from_value(values.get("resolver_config"))
    values["cache_config"] = CacheServiceConfig.from_value(values.get("cache_config"))
    return TrainConfig(**values)


def model_config_from_dict(raw: dict, vocab_size: int, train_cfg: TrainConfig) -> PRAConfig:
    model = raw.get("model", {})
    pra = raw.get("pra", model)
    return PRAConfig(
        vocab_size=vocab_size,
        d_model=int(model.get("d_model", 128)),
        n_heads=int(model.get("n_heads", 4)),
        n_layers=int(model.get("n_layers", 4)),
        n_vanilla_layers=int(model.get("n_vanilla_layers", 0)),
        n_mixed_layers=int(model.get("n_mixed_layers", 0)),
        max_seq_len=train_cfg.max_seq_len,
        dropout=float(model.get("dropout", 0.0)),
        pra_layer_ids=tuple(pra.get("pra_layer_ids", (2, 3))),
        top_k_refs=int(pra.get("top_k_refs", 2)),
        trigger_threshold=float(pra.get("trigger_threshold", 0.2)),
        use_cross_attention_memory=bool(pra.get("use_cross_attention_memory", True)),
        use_concat_memory=bool(pra.get("use_concat_memory", False)),
        memory_alpha=float(pra.get("memory_alpha", 0.5)),
        batch_size=train_cfg.batch_size,
        lr=train_cfg.learning_rate,
        steps=train_cfg.epochs,
        device=train_cfg.device,
    )


def build_trainer(
    config_path: str,
    output_dir: str | None = None,
    resume_from: str | None = None,
    device: str | None = None,
    model_name: str = "standalone_tiny",
):
    raw = load_config(config_path, model_name=model_name)
    train_cfg = train_config_from_dict(raw)
    if output_dir:
        train_cfg.output_dir = output_dir
    if resume_from:
        train_cfg.resume_from = resume_from
    if device:
        train_cfg.device = device
    dm = PRADataModule(
        dataset_stage=train_cfg.dataset_stage,
        data_dir=train_cfg.data_dir,
        max_examples=train_cfg.max_examples,
        batch_size=train_cfg.batch_size,
        max_seq_len=train_cfg.max_seq_len,
        shuffle=train_cfg.shuffle,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        persistent_workers=train_cfg.persistent_workers,
    ).load()
    model_cfg = model_config_from_dict(raw, dm.tokenizer.vocab_size, train_cfg)
    return PRAStandaloneTrainer(model_cfg, train_cfg, dm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--model", default="standalone_tiny")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--device")
    args = parser.parse_args()
    trainer = build_trainer(args.config, args.output_dir, args.resume_from, args.device, args.model)
    trainer.train()


if __name__ == "__main__":
    main()
