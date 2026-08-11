"""Thin Hugging Face adapters backed by the shared PRA execution core."""

from .config import PRAHFConfig
from .injection import PRAHFModel, inject_pra

__all__ = ["PRAHFConfig", "PRAHFModel", "inject_pra"]
