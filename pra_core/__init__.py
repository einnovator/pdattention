from pathlib import Path

_src_package = Path(__file__).resolve().parent.parent / "src" / "pra_core"
__path__.insert(0, str(_src_package))

from .datasets import DatasetExample, load_dataset
from .references import ReferenceHandle, ReferenceTable, ResolvedReference

__all__ = ["DatasetExample", "ReferenceHandle", "ReferenceTable", "ResolvedReference", "load_dataset"]
