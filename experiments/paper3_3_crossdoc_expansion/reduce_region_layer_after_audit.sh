#!/bin/sh
set -eu

ROOT=${PAPER3_3_ROOT:-/Users/jorge.simao/git/rd/pdattention-paper3-3-factorial}
PYTHON=${PAPER3_3_PYTHON:-/Users/jorge.simao/git/rd/pdattention-crossdoc/.venv/bin/python}
RUN_DIR=${PAPER3_3_RUN_DIR:-$ROOT/.runs/paper3_3_region_layer_test_n150}
OUTPUT=${PAPER3_3_OUTPUT:-$RUN_DIR/publication}

# The generation runner writes summary.json only after every checkpoint has
# been consolidated. Refuse to reduce a partial or failed cohort.
while [ ! -f "$RUN_DIR/summary.json" ]; do
  if ! pgrep -f 'experiments.paper3_3_crossdoc_expansion.run_expansion_generation.*paper3_3_region_layer_test_n150' >/dev/null; then
    echo "Region/layer audit stopped without producing $RUN_DIR/summary.json" >&2
    exit 2
  fi
  sleep 60
done

cd "$ROOT"
PYTHONPATH=src:. "$PYTHON" -m experiments.paper3_3_crossdoc_expansion.summarize_region_layer \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT"
