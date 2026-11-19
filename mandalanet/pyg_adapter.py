"""Optional interoperability with PyTorch Geometric.

The critical path is plain PyTorch (works anywhere, no compiled extras). If a
PyG install is available, this module exposes the same layers as
`torch_geometric.nn.MessagePassing`/`Data` so the model can live inside a
normal PyG pipeline. Parity is exact: the wrapper feeds PyG's tensors into the
pure-torch layer instead of re-deriving the math.
"""
from __future__ import annotations

import torch

from .graph import Graph
from .layers import RadialAttention, RadialBlock

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing

    HAS_PYG = True
except Exception:  # pragma: no cover
    Data = None  # type: ignore
    MessagePassing = object  # type: ignore
    HAS_PYG = False


def to_data(g: Graph):
    if not HAS_PYG:
        raise ImportError("torch_geometric is not installed")
    kwargs = dict(x=g.x, edge_index=g.edge_index, pos=g.pos, edge_type=g.edge_type,
                  batch=g.batch, num_nodes=g.num_nodes)
    return Data(**kwargs)


def from_data(d) -> Graph:
    return Graph(x=d.x, pos=d.pos, edge_index=d.edge_index,
                 edge_type=getattr(d, "edge_type", None),
                 batch=getattr(d, "batch", None))


class GeoConv(MessagePassing):
    """PyG-shaped wrapper around `RadialAttention`."""

    def __init__(self, layer: RadialAttention, edge_feat_dim: int):
        super().__init__(aggr="add")
        self.layer = layer
        self.edge_feat_dim = edge_feat_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        g = Graph(x=x, pos=torch.zeros((x.shape[0], 2), dtype=x.dtype,
                                       device=x.device), edge_index=edge_index)
        return self.layer(x, g, edge_attr)[0]

    def message(self, x_j, edge_attr):
        return x_j  # unused: the wrapper delegates to the pure-torch layer


def wrap(block: RadialBlock, edge_feat_dim: int):
    """Return (MessagePassing conv, wrapped block) sharing weights with `block`."""
    conv = GeoConv(block.attn, edge_feat_dim)
    return conv
