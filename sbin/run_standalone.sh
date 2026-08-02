#!/usr/bin/env bash
set -euo pipefail
python -m pra_torch.train --steps 200 --batch-size 4 --out pra_tiny.pt
python -m pra_torch.eval --ckpt pra_tiny.pt
pytest -q
