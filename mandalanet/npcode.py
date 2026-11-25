"""The same weights, sampled in NumPy.

Step 4 measured why PyTorch is the wrong runtime for one-token-at-a-time
decoding on this device: a (1, 96) x (96, 288) matmul costs 400-1400 us through
oneDNN and ~24 us without it, and a single cached step makes ~500 aten calls,
so the 385 k model generates 47 B/s. The arithmetic is not the problem -- 24 us
per matmul is fine -- the per-op dispatch is.

So this module exports a trained CodeLM to a small .npz and decodes it with
18 numpy matmuls per byte. It is deliberately a *copy* of the architecture, and
is checked against torch in tests: same logits to float32 rounding, same greedy
bytes. Training stays in torch; only sampling moves.

    from mandalanet.npcode import export, NumpyLM
    export("runs/budget/code-rope.pt", "runs/budget/code-rope.npz")
    lm = NumpyLM("runs/budget/code-rope.npz")
    lm.generate([b"def mean(xs):\\n    "], 32, temp=None)

Weights are stored int8 (per-row symmetric) and dequantised once at load, so
the file on disk is a quarter of the fp32 size. Dequantisation is not free but
it happens once, not per byte.
"""
from __future__ import annotations

import math

import numpy as np

LN_EPS = 1e-5
TOP_P = 0.95


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _quantise(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8. Returns (q, scale) with w ~ q * scale[:, None]."""
    m = np.abs(w).max(axis=1, keepdims=True)
    scale = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def export(ckpt: str, path: str, int8: bool = True) -> dict:
    """Turn a CodeLM checkpoint into an .npz the numpy sampler can load."""
    import torch

    from mandalanet.code import CodeLM

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["pos"] == "rope", "the numpy sampler implements the rope arm only"
    model = CodeLM(**cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    def arr(t: "torch.Tensor") -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    sd = model.state_dict()
    out = {"cfg": np.array([cfg["dim"], cfg["depth"], cfg["heads"],
                            cfg["ctx"], cfg["vocab"]], dtype=np.int64)}

    def put(name: str, w: np.ndarray) -> None:
        if int8 and w.ndim == 2:                       # linear weights only
            q, s = _quantise(w)
            out[name] = q
            out[name + ".scale"] = s
        else:
            out[name] = w.astype(np.float32)

    put("tok", arr(sd["tok.weight"]))
    for i in range(cfg["depth"]):
        p = f"b{i}."
        put(p + "ln1_w", arr(sd[f"blocks.{i}.ln1.weight"]))
        put(p + "ln1_b", arr(sd[f"blocks.{i}.ln1.bias"]))
        put(p + "qkv_w", arr(sd[f"blocks.{i}.qkv.weight"]))
        put(p + "qkv_b", arr(sd[f"blocks.{i}.qkv.bias"]))
        put(p + "out_w", arr(sd[f"blocks.{i}.out.weight"]))
        put(p + "out_b", arr(sd[f"blocks.{i}.out.bias"]))
        put(p + "ln2_w", arr(sd[f"blocks.{i}.ln2.weight"]))
        put(p + "ln2_b", arr(sd[f"blocks.{i}.ln2.bias"]))
        put(p + "ff1_w", arr(sd[f"blocks.{i}.ff.0.weight"]))
        put(p + "ff1_b", arr(sd[f"blocks.{i}.ff.0.bias"]))
        put(p + "ff2_w", arr(sd[f"blocks.{i}.ff.3.weight"]))
        put(p + "ff2_b", arr(sd[f"blocks.{i}.ff.3.bias"]))
    put("lnf_w", arr(sd["ln_f.weight"]))
    put("lnf_b", arr(sd["ln_f.bias"]))
    put("head_w", arr(sd["head.weight"]))
    put("head_b", arr(sd["head.bias"]))

    np.savez(path, **out)
    return {"path": path, "int8": int8, "config": cfg,
            "params": model.n_params()}


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf, Abramowitz-Stegun 7.1.26 (max abs error 1.5e-7).

    math.erf is scalar, and a Python-level call per element costs more than the
    whole matmul; this keeps the sampler in numpy ufuncs.
    """
    a = np.abs(x)
    t = a * 0.3275911
    t += 1.0
    np.reciprocal(t, out=t)
    p = t * np.float32(1.061405429)                       # Horner, five terms
    p += np.float32(-1.453152027)
    p *= t
    p += np.float32(1.421413741)
    p *= t
    p += np.float32(-0.284496736)
    p *= t
    p += np.float32(0.254829592)
    p *= t
    p *= np.exp(-a * a)
    p = 1.0 - p
    np.copysign(p, x, out=p)
    return p


def _gelu(x: np.ndarray) -> np.ndarray:
    # torch's default GELU is the exact erf form
    return 0.5 * x * (1.0 + _erf(x * (1.0 / math.sqrt(2.0))))


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
class NumpyLM:
    _ones: dict[int, np.ndarray] = {}

    def __init__(self, path: str):
        z = np.load(path)
        dim, depth, heads, ctx, vocab = (int(v) for v in z["cfg"])
        self.dim, self.depth, self.heads, self.ctx, self.vocab = (
            dim, depth, heads, ctx, vocab)
        self.dh = dim // heads

        def w(name: str) -> np.ndarray:
            a = z[name]
            return a.astype(np.float32) * z[name + ".scale"] if a.dtype == np.int8 else a

        self.tok = w("tok")
        self.blocks = []
        for i in range(depth):
            p = f"b{i}."

            def folded(ln_w: np.ndarray, ln_b: np.ndarray,
                       W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                """LayerNorm's affine, folded into the linear that follows it.

                ((x-mu)*S*gamma + beta) @ W.T + b
                  = ((x-mu)*S) @ (gamma[:,None]*W).T + (beta @ W.T + b)
                which turns two elementwise ops per layer norm into a
                precomputed matrix and bias.
                """
                return (W * ln_w[None, :]).T.copy(), b + ln_b @ W.T

            qkv_gw, qkv_gb = folded(w(p + "ln1_w"), w(p + "ln1_b"),
                                    w(p + "qkv_w"), w(p + "qkv_b"))
            ff1_gw, ff1_gb = folded(w(p + "ln2_w"), w(p + "ln2_b"),
                                    w(p + "ff1_w"), w(p + "ff1_b"))
            self.blocks.append({
                "qkv_gw": qkv_gw, "qkv_gb": qkv_gb,
                "ff1_gw": ff1_gw, "ff1_gb": ff1_gb,
                "out_w": w(p + "out_w").T.copy(), "out_b": w(p + "out_b"),
                "ff2_w": w(p + "ff2_w").T.copy(), "ff2_b": w(p + "ff2_b"),
            })
        self.head_gw, self.head_gb = folded(w("lnf_w"), w("lnf_b"),
                                            w("head_w"), w("head_b"))

        # RoPE as one small matmul per call. The rotary matrix is block
        # diagonal with 2x2 rotations, so (B,H,n,dh) @ R[offset:offset+n]
        # reproduces the even/odd formula in one gemm instead of six ufuncs.
        inv = 1.0 / (10000.0 ** (np.arange(0, self.dh, 2, dtype=np.float32) / self.dh))
        f = np.outer(np.arange(ctx, dtype=np.float32), inv)
        cos, sin = np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)
        self.rot = np.zeros((ctx, self.dh, self.dh), dtype=np.float32)
        idx = np.arange(self.dh)
        self.rot[:, idx, idx] = 1.0
        e, o = idx[0::2], idx[1::2]
        self.rot[:, e, e] = cos
        self.rot[:, o, o] = cos
        self.rot[:, e, o] = sin                           # y_e = c*x_e - s*x_o
        self.rot[:, o, e] = -sin                          # y_o = s*x_e + c*x_o
        self.rot_q = self.rot * np.float32(1.0 / math.sqrt(self.dh))
        ones = self._ones.setdefault(dim, np.full((dim, 1), 1.0 / dim, dtype=np.float32))
        assert ones.shape == (dim, 1)

    # -- primitives ---------------------------------------------------------
    @staticmethod
    def _ln_mm(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """LayerNorm (affine folded into W) followed by the matmul.

        numpy's mean/var along the last axis cost 19/30 us here while the same
        sum as a (dim, 1) matmul costs 2.5 us, and every elementwise op costs
        ~2.8 us whatever its size, so the op count is what matters.
        """
        ones = NumpyLM._ones[x.shape[-1]]
        mu = x @ ones                                     # (B, n, 1)
        xc = x - mu
        var = (xc * xc) @ ones
        scale = np.sqrt(var + LN_EPS)
        np.reciprocal(scale, out=scale)
        xc *= scale
        return xc @ W + b

    def _step(self, idx: np.ndarray, state):
        """One call for n new tokens. state holds the k/v buffers and length."""
        b, n = idx.shape
        h, dh = self.heads, self.dh
        offset = state["len"]
        total = offset + n
        assert total <= self.ctx, "window full"
        x = self.tok[idx]                                    # (B, n, dim)

        # causal mask for the new tokens against the whole window
        if n == 1:
            mask = None
        else:
            mask = np.zeros((n, total), dtype=np.float32)
            tri = np.triu(np.full((n, n), -np.inf, dtype=np.float32), k=1)
            mask[:, offset:offset + n] = tri                  # token i sees j <= offset+i

        rot, rot_q = self.rot[offset:total], self.rot_q[offset:total]
        for i, blk in enumerate(self.blocks):
            qkv = self._ln_mm(x, blk["qkv_gw"], blk["qkv_gb"])  # (B, n, 3*dim)
            qkv = qkv.reshape(b, n, 3, h, dh).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = np.einsum("bhnk,nke->bhne", q, rot_q)
            k = np.einsum("bhnk,nke->bhne", k, rot)

            kb, vb = state["k"][i], state["v"][i]             # (B, H, ctx, dh)
            kb[:, :, offset:total] = k
            vb[:, :, offset:total] = v
            kk, vv = kb[:, :, :total], vb[:, :, :total]

            scores = q @ kk.transpose(0, 1, 3, 2)             # (B, H, n, total)
            if mask is not None:
                scores += mask
            scores -= scores.max(-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(-1, keepdims=True)
            a = scores @ vv                                   # (B, H, n, dh)
            a = a.transpose(0, 2, 1, 3).reshape(b, n, self.dim)

            x += a @ blk["out_w"] + blk["out_b"]
            x += _gelu(self._ln_mm(x, blk["ff1_gw"], blk["ff1_gb"])) @ blk["ff2_w"] \
                + blk["ff2_b"]

        state["len"] = total
        return self._ln_mm(x, self.head_gw, self.head_gb)

    def _state(self, batch: int):
        return {"len": 0,
                "k": [np.zeros((batch, self.heads, self.ctx, self.dh), dtype=np.float32)
                      for _ in range(self.depth)],
                "v": [np.zeros((batch, self.heads, self.ctx, self.dh), dtype=np.float32)
                      for _ in range(self.depth)]}

    # -- public -------------------------------------------------------------
    def logits(self, idx: np.ndarray) -> np.ndarray:
        """Full forward for a (B, L) prompt; returns (B, L, vocab)."""
        state = self._state(idx.shape[0])
        return self._step(idx, state)

    def generate(self, prompts: list[bytes], n_new: int, temp: float | None = None,
                 seed: int = 0, top_p: float = TOP_P) -> list[bytes]:
        """Left-pad the prompts and decode; mirrors tools/gen_meter.generate."""
        rng = np.random.default_rng(seed)
        L = max(len(p) for p in prompts)
        idx = np.full((len(prompts), L), 10, dtype=np.int64)
        for i, p in enumerate(prompts):
            idx[i, L - len(p):] = np.frombuffer(p, dtype=np.uint8)
        state = self._state(idx.shape[0])
        logits = self._step(idx, state)
        outs = [[] for _ in prompts]
        for _ in range(n_new):
            step = logits[:, -1].astype(np.float64)
            if temp is None:
                nxt = step.argmax(-1)
            else:
                p = np.exp(step / temp - (step / temp).max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                nxt = np.empty(len(prompts), dtype=np.int64)
                for row in range(len(prompts)):
                    order = np.argsort(-p[row])
                    cum = np.cumsum(p[row][order])
                    keep = order[:int(np.searchsorted(cum, top_p)) + 1]
                    probs = p[row][keep] / p[row][keep].sum()
                    nxt[row] = rng.choice(keep, p=probs)
            for i in range(len(prompts)):
                outs[i].append(int(nxt[i]))
            logits = self._step(nxt[:, None], state)
        return [bytes(o) for o in outs]
