"""mandala-gnn: geometric GNN over hand-drawn mandalas.

The names below are imported lazily, so `python -m mandalanet.serve` does not
pay for torch at startup: the demo page comes up first, and its launch button
is what imports torch and reads the weights.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "Graph", "EdgeGeometry", "batch_graphs", "build_mesh", "MESHES",
    "MandalaNet", "ChaosPyramid", "RadialAttention", "RadialBlock",
]
__version__ = "0.1.0"

_MODULES = {
    "Graph": "graph",
    "EdgeGeometry": "graph",
    "batch_graphs": "graph",
    "build_mesh": "lattices",
    "MESHES": "lattices",
    "MandalaNet": "model",
    "ChaosPyramid": "pyramid",
    "RadialAttention": "layers",
    "RadialBlock": "layers",
}


def __getattr__(name: str) -> Any:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
