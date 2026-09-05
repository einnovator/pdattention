from __future__ import annotations

import argparse
import json

import pytest

from experiments.paper3_2_rag.run_crossdoc_precision_campaign import (
    _complete,
    _seeds,
)


def test_seed_parser_and_resume_contract(tmp_path) -> None:
    assert _seeds("11, 23,37") == (11, 23, 37)
    with pytest.raises(argparse.ArgumentTypeError, match="at least one seed"):
        _seeds(" , ")
    manifest = tmp_path / "manifest.json"
    assert not _complete(manifest)
    manifest.write_text(
        json.dumps({"completed_unix": 123.0, "rows": 30}), encoding="utf-8"
    )
    assert _complete(manifest)
