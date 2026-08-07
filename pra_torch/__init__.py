import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent / "src"
_src_package = _src_root / "pra_torch"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))
__path__.insert(0, str(_src_package))
