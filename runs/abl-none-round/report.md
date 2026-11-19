# Ablations

2 seeds x 400 steps, drawing `drawings/hex-tessellation.jpg`, mean +- sd over seeds.

`val` = held-out cells, `wedge` = one 45 deg sector never trained on. The do-nothing mean-patch predictor scores val 0.0926, wedge 0.0228 MSE.

## mesh

| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |
|---|---|---|---|---|---|---|
| none | 0.01498 +- 0.00239 | 0.02165 +- 0.00070 | 0.882 +- 0.001 | 405342 | 196 | 0.000 |

## How to read this

- `mesh=none` is the same network with every edge deleted: an MLP.
- `jitter`/`random` keep node count and degree but break the honeycomb.
- `harmonics=0` removes the angular symmetry bias, `edge_gate=False`
  removes topology adaptation, `depth=1` removes the pyramid.
