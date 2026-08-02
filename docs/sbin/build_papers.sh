#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
papers_root="$repo_root/docs/papers"

if [ "$#" -eq 0 ]; then
  papers=(paper0_position paper1_standalone_pra)
else
  papers=("$@")
fi

for paper in "${papers[@]}"; do
  paper_dir="$papers_root/$paper"
  if [ ! -f "$paper_dir/paper.tex" ]; then
    echo "Missing paper.tex for $paper at $paper_dir" >&2
    exit 1
  fi
  (cd "$paper_dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex)
done
