# Ablations

2 seeds x 400 steps, drawing `drawings/hex-tessellation.jpg`, mean +- sd over seeds.

`val` = held-out cells, `wedge` = one 45 deg sector never trained on. `mean-patch` is the do-nothing predictor: val 0.0224, wedge 0.0282.

## mesh

| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |
|---|---|---|---|---|---|---|
| hex | 0.00121 +- 0.00037 | 0.00198 +- 0.00089 | 0.989 +- 0.005 | 405342 | 203 | 0.278 |
| rewired | 0.00424 +- 0.00037 | 0.00246 +- 0.00072 | 0.987 +- 0.004 | 405342 | 203 | 0.245 |
| grid | 0.00257 +- 0.00122 | 0.00193 +- 0.00003 | 0.990 +- 0.000 | 405342 | 196 | 0.338 |
| none | 0.01498 +- 0.00239 | 0.02165 +- 0.00070 | 0.882 +- 0.001 | 405342 | 196 | 0.000 |

## How to read this

- `mesh=none` is the same network with every edge deleted: an MLP.
- `jitter`/`random` keep node count and degree but break the honeycomb.
- `harmonics=0` removes the angular symmetry bias, `edge_gate=False`
  removes topology adaptation, `depth=1` removes the pyramid.
