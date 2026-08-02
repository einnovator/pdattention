from pathlib import Path

_src_package = Path(__file__).resolve().parent.parent / "src" / "data"
__path__.insert(0, str(_src_package))
