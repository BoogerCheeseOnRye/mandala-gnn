"""Write the cold-state cache the demo server serves before torch is loaded.

    python tools/dump_demo_state.py [--model runs/main-hex/model.pt]

Writes `demo.json` next to the checkpoint: the drawing, node/edge positions,
masks and config. The server then starts in about a second; the page's launch
button pays for the torch import. Run this again after retraining or changing
the drawing; the server checks the node count and refuses a stale cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mandalanet.serve import Runner  # noqa: E402

KEEP = ["config", "metrics", "nodes", "edges", "wedge", "eval",
        "val_sector", "size", "drawing", "img_url", "sectors"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/main-hex/model.pt")
    p.add_argument("--out", default="")
    a = p.parse_args()

    model = os.path.abspath(a.model)
    runner = Runner(model, force_build=True)
    st = runner.state()
    demo = {k: st[k] for k in KEEP}
    demo["nodes_n"] = runner.ex.graph.num_nodes
    demo["edges_n"] = runner.ex.graph.num_edges
    demo["patch_dim"] = int(runner.ex.patch_dim)

    out = a.out or os.path.join(os.path.dirname(model), "demo.json")
    with open(out, "w") as f:
        json.dump(demo, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB): "
          f"{demo['nodes_n']} nodes, {demo['edges_n']} edges, "
          f"{len(demo['eval'])} eval cells, {len(demo['wedge'])} held out")


if __name__ == "__main__":
    main()
