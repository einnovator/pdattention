from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_standalone import build_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--model", default="standalone_tiny")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--predictions-jsonl")
    parser.add_argument("--traces-jsonl")
    args = parser.parse_args()
    trainer = build_trainer(args.config, args.output_dir, args.checkpoint, args.device, args.model)
    pred_path = args.predictions_jsonl
    trace_path = args.traces_jsonl
    if trace_path is None:
        trace_path = str(Path(trainer.trace_dir) / "eval_traces.jsonl")
    trainer.test(save_predictions=pred_path, save_traces=trace_path)


if __name__ == "__main__":
    main()
