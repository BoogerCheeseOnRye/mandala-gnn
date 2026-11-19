"""Local demo server: the trained model, live, in a browser.

    python -m mandalanet.serve --model runs/main-hex/model.pt --port 8000

Serves web/ and three endpoints:

    GET  /api/state     model config, metrics, ablation tables, mesh, drawing
    POST /api/load      build the network from the checkpoint (idempotent)
    POST /api/redraw    {"hidden": [node ids]} -> prediction, per-edge gates

Nothing heavy is imported at startup. The drawing, mesh and metrics come from
`demo.json` next to the checkpoint, so the page opens in about a second. The
page's "launch model" button imports torch, builds the network and reads the
weights; redraw does the same if it is called first. `tools/dump_demo_state.py`
regenerates the cache when the checkpoint or drawing changes.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
IMAGE = 128


def png(t: Any, scale: int = 1) -> str:
    arr = (t.detach().squeeze().clamp(0, 1) * 255).byte().numpy()
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def read_json(path: str) -> dict:
    try:
        with open(os.path.join(ROOT, path)) as f:
            return json.load(f)
    except Exception:
        return {}


def read_json_at(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


class Runner:
    """One checkpoint, one drawing. Cold until the page asks for the model."""

    def __init__(self, model_path: str, preload: bool = False,
                 force_build: bool = False):
        self.model_path = os.path.abspath(model_path)
        self.lock = threading.RLock()
        self.loaded = False
        self.model = None
        self.load_ms = None
        self.ex = None
        self.cfg = None

        base = os.path.dirname(self.model_path)
        meta = read_json_at(os.path.join(base, "metrics.json"))
        self.cfg_raw = dict(meta.get("config") or {})
        self.metrics = meta.get("metrics") or {}
        self.history = meta.get("history") or []

        # The state cache holds the drawing, mesh and mask the page draws before
        # torch is anywhere in memory. It is optional: without it the example is
        # built at startup, which costs a torch import (~4 s on a phone).
        self.demo = ({} if force_build
                     else read_json_at(os.path.join(base, "demo.json")))
        if self.demo:
            self.cfg_raw = self.cfg_raw or dict(self.demo.get("config") or {})
            self.img_url = self.demo["drawing"]
        else:
            self._build_example()
            self.img_url = png(self.ex.img, scale=3)
        if preload:
            self.load()

    # ------------------------------------------------------------------ #
    def _build_example(self) -> None:
        """Build the drawing, mesh and masks. Pulls in torch."""
        import torch

        from .data import recon_example
        from .train import Config

        if not self.cfg_raw:
            ckpt = torch.load(self.model_path, map_location="cpu",
                              weights_only=False)
            self.cfg_raw = dict(ckpt.get("config") or {})
        fields = Config.__dataclass_fields__
        cfg = Config(**{k: v for k, v in self.cfg_raw.items() if k in fields})
        cfg.threads = 1
        torch.set_num_threads(1)
        self.cfg = cfg
        self.ex = recon_example(cfg.drawing, mesh=cfg.mesh, seed=cfg.seed,
                                circle=cfg.round_patches)

    def _ensure_example(self) -> None:
        if self.ex is None:
            self._build_example()
        if self.demo:
            want = self.demo.get("nodes_n")
            got = self.ex.graph.num_nodes
            if want and want != got:
                raise RuntimeError(
                    f"demo.json has {want} nodes but the drawing builds {got}; "
                    "regenerate it with tools/dump_demo_state.py")

    def load(self) -> None:
        """Read the weights and warm the kernels. Safe to call repeatedly."""
        with self.lock:
            if self.loaded:
                return
            t0 = time.time()
            import torch

            from .data import with_mask
            from .train import build

            self._ensure_example()
            ckpt = torch.load(self.model_path, map_location="cpu",
                              weights_only=False)
            model = build(self.cfg, self.ex.patch_dim + 1, self.ex.patch_dim)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            # one empty-mask pass: the first click should not pay for lazy
            # kernel initialisation
            with torch.no_grad():
                model(with_mask(self.ex, torch.zeros(
                    self.ex.graph.num_nodes, dtype=torch.bool)))
            self.model = model
            self.metrics = ckpt.get("metrics") or self.metrics
            self.load_ms = round((time.time() - t0) * 1000)
            self.loaded = True

    # ------------------------------------------------------------------ #
    @property
    def num_nodes(self) -> int:
        if self.ex is not None:
            return self.ex.graph.num_nodes
        return int(self.demo.get("nodes_n", 0))

    @property
    def num_edges(self) -> int:
        if self.ex is not None:
            return self.ex.graph.num_edges
        return int(self.demo.get("edges_n", 0))

    def state(self) -> dict:
        if self.ex is not None:
            g = self.ex.graph
            pos = self.ex.mesh.pos
            px = (pos * 0.5 + 0.5) * IMAGE
            nodes = [[round(float(x), 2), round(float(y), 2)] for x, y in px]
            edges = [[int(s), int(d), int(t)] for (s, d), t in
                     zip(g.edge_index.t().tolist(), g.edge_type.tolist())]
            wedge = [int(i) for i in
                     self.ex.wedge_mask.nonzero().flatten().tolist()]
            evaluation = [int(i) for i in
                          (self.ex.train_mask | self.ex.val_mask |
                           self.ex.wedge_mask).nonzero().flatten().tolist()]
            sectors = int(self.cfg.n_sectors)
            val_sector = int(self.cfg.val_sector)
            size = IMAGE
        else:
            d = self.demo
            nodes, edges = d["nodes"], d["edges"]
            wedge, evaluation = d["wedge"], d["eval"]
            sectors = int(d["sectors"])
            val_sector = int(d["val_sector"])
            size = int(d["size"])

        abl_hex = read_json("runs/abl-hex400/results.json").get("table", {})
        abl_mod = read_json("runs/abl-model/results.json").get("table", {})
        abl_round = read_json("runs/abl-round/results.json").get("table", {})
        mlp = [read_json(f"runs/mlp-{s}/metrics.json").get("metrics", {})
               for s in (0, 1)]
        return {
            "config": {k: v for k, v in self.cfg_raw.items() if k != "out"},
            "metrics": self.metrics,
            "loaded": self.loaded,
            "load_ms": self.load_ms,
            "model": os.path.basename(self.model_path),
            "checkpoint_kb": round(os.path.getsize(self.model_path) / 1024),
            "nodes": nodes,
            "edges": edges,
            "wedge": wedge,
            "eval": evaluation,
            "val_sector": val_sector,
            "size": size,
            "drawing": self.img_url,
            "img_url": self.img_url,
            "abl_mesh": abl_hex.get("mesh", {}),
            "abl_harmonics": abl_mod.get("harmonics", {}),
            "abl_depth": abl_mod.get("depth", {}),
            "abl_gate": abl_mod.get("edge_gate", {}),
            "abl_round": abl_round.get("mesh", {}),
            "mlp": mlp,
            "history": self.history,
            "sectors": sectors,
        }

    # ------------------------------------------------------------------ #
    def redraw(self, hidden_ids: list[int]) -> dict:
        self.load()
        import torch

        from .data import patches_to_image, with_mask
        from .model import masked_corr, masked_mse

        n = self.ex.graph.num_nodes
        hidden = torch.zeros(n, dtype=torch.bool)
        ids = [int(i) for i in hidden_ids if 0 <= int(i) < n]
        if ids:
            hidden[torch.tensor(sorted(set(ids)))] = True
        t0 = time.time()
        with self.lock, torch.no_grad():
            g = with_mask(self.ex, hidden)
            out = self.model(g)
            pred = out["node_pred"]
            pred_img = patches_to_image(pred, self.ex.mesh.pos_px, IMAGE)
            mask_img = patches_to_image(hidden.float().unsqueeze(-1),
                                        self.ex.mesh.pos_px, IMAGE)
            composite = torch.where(mask_img > 0.5, pred_img, self.ex.img)

            # The gates are (edges, heads): average over the fine-level
            # blocks for the canvas, but keep each edge's dominant head.
            fine_gates = [gt for gt in out["gates"]
                          if gt.numel() and gt.shape[0] == g.num_edges]
            if fine_gates:
                per_head_edge = torch.stack(fine_gates, 0).mean(0)  # (E,H)
                gates = per_head_edge.mean(1)
                head_edge = per_head_edge.argmax(1)
            else:
                gates = torch.zeros(0)
                head_edge = torch.zeros(0, dtype=torch.long)

            # The gate is a function of edge geometry, so it is the same on
            # every pass. The attention weights are not: they depend on what
            # is visible, which is what the live head meters show.
            fine_alpha = [a for a in out.get("alphas", [])
                          if a.numel() and a.shape[0] == g.num_edges]
            if fine_alpha:
                a_mean = torch.stack(fine_alpha, 0).mean(0)          # (E,H)
                share = a_mean / (a_mean.sum(-1, keepdim=True) + 1e-8)
                head_attn = share.mean(0)                            # sums to 1
                head_edge = share.argmax(1)
                activity = a_mean.mean(1)
            else:
                head_attn = (per_head_edge.mean(0) if fine_gates
                             else torch.zeros(0))
                activity = gates
            per_node = ((pred - self.ex.target) ** 2).mean(1)
            wedge = self.ex.wedge_mask
            wedge_full = bool(hidden[wedge].all())
            stats = {
                "wedgeMse": (float(masked_mse(pred, self.ex.target, wedge))
                             if wedge_full else None),
                "wedgeCorr": (float(masked_corr(pred, self.ex.target, wedge))
                              if wedge_full else None),
                "cells": int(hidden.sum()),
                "mse": float(masked_mse(pred, self.ex.target, hidden)) if ids else 0.0,
                "corr": float(masked_corr(pred, self.ex.target, hidden)) if ids else 0.0,
                "gate": float(out["gate_mean"]),
                "levels": [round(float(w), 4) for w in out["level_weights"]],
                "depth": int(out["depth_used"]),
                "ms": round((time.time() - t0) * 1000),
                "nodes": n,
                "edges": int(g.num_edges),
            }
            hops = self._hops(hidden)
            return {
                "composite": png(composite, scale=3),
                "pred": png(pred_img, scale=3),
                "hidden": sorted(set(ids)),
                "gates": [round(float(x), 4) for x in gates],
                "nodeErr": [round(float(x), 5) for x in per_node],
                "hop": hops,
                "heads": [round(float(x), 4) for x in head_attn],
                "headEdge": [int(x) for x in head_edge],
                "stats": stats,
                "top": self._top_edges(activity),
                "headsSum": 1.0 if fine_alpha else 0.0,
            }

    def _hops(self, hidden) -> list[int]:
        """BFS distance from any visible cell: the order a message has to travel."""
        n = self.ex.graph.num_nodes
        adj: list[list[int]] = [[] for _ in range(n)]
        for s, d in self.ex.graph.edge_index.t().tolist():
            adj[s].append(d)
        dist = [-1] * n
        q = [i for i in range(n) if not bool(hidden[i])]
        for i in q:
            dist[i] = 0
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return [min(d, 9) if d >= 0 else 9 for d in dist]

    @staticmethod
    def _top_edges(gates) -> list[int]:
        if gates.numel() == 0:
            return []
        k = min(80, gates.numel())
        return [int(i) for i in gates.topk(k).indices.tolist()]


class Handler(BaseHTTPRequestHandler):
    runner: Runner = None  # type: ignore
    quiet = True

    def log_message(self, fmt, *args):  # noqa: A003
        if not self.quiet:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ #
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._file("style.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._file("app.js", "application/javascript; charset=utf-8")
        if path == "/icon.png":
            return self._file("icon.png", "image/png")
        if path == "/recon.png":
            model_dir = os.path.dirname(self.runner.model_path)
            return self._png(os.path.join(model_dir, "recon.png"))
        if path == "/api/state":
            try:
                return self._json(self.runner.state())
            except Exception as err:  # noqa: BLE001
                return self._json({"error": str(err)}, 500)
        if path == "/favicon.ico":
            return self._send(b"", "image/x-icon", 204)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path not in ("/api/redraw", "/api/load"):
            return self._json({"error": "not found"}, 404)
        try:
            if path == "/api/load":
                self.runner.load()
                return self._json(self.runner.state())
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            hidden = body.get("hidden", [])
            return self._json(self.runner.redraw(hidden))
        except Exception as err:  # noqa: BLE001
            return self._json({"error": str(err)}, 500)

    def _png(self, path: str) -> None:
        if not os.path.exists(path):
            return self._json({"error": "no recon.png next to the checkpoint"}, 404)
        with open(path, "rb") as f:
            self._send(f.read(), "image/png")

    def _file(self, name: str, ctype: str) -> None:
        path = os.path.join(WEB, name)
        if not os.path.exists(path):
            return self._json({"error": f"missing {name}"}, 404)
        with open(path, "rb") as f:
            self._send(f.read(), ctype)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/main-hex/model.pt")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--preload", action="store_true",
                   help="load the checkpoint at startup instead of on demand")
    a = p.parse_args()

    runner = Runner(a.model, preload=a.preload)
    Handler.runner = runner
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"mandala-gnn demo on http://localhost:{a.port}  "
          f"({runner.num_nodes} nodes, {runner.num_edges} edges, "
          f"{'model loaded' if runner.loaded else 'model cold: launch it from the page'})",
          flush=True)
    print("ctrl-c to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
