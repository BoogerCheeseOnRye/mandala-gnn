# Ablations

2 seeds x 300 steps, drawing `drawings/hex-tessellation.jpg`, mean +- sd over seeds.

`val` = held-out cells, `wedge` = one 45 deg sector never trained on. The do-nothing mean-patch predictor scores val 0.0890, wedge 0.0148 MSE.

## harmonics

| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |
|---|---|---|---|---|---|---|
| 2 | 0.00254 +- 0.00162 | 0.00340 +- 0.00004 | 0.870 +- 0.001 | 401886 | 203 | 0.301 |
| 0 | 0.03942 +- 0.00670 | 0.01283 +- 0.00181 | 0.238 +- 0.319 | 398430 | 203 | 0.350 |
| 4 | 0.00288 +- 0.00042 | 0.00324 +- 0.00077 | 0.888 +- 0.014 | 405342 | 203 | 0.327 |
| 8 | 0.00228 +- 0.00051 | 0.00265 +- 0.00014 | 0.901 +- 0.001 | 412254 | 203 | 0.252 |

## depth

| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |
|---|---|---|---|---|---|---|
| 1 | 0.00268 +- 0.00075 | 0.00408 +- 0.00028 | 0.844 +- 0.014 | 148012 | 203 | 0.386 |
| 2 | 0.00218 +- 0.00005 | 0.00334 +- 0.00040 | 0.877 +- 0.019 | 276677 | 203 | 0.388 |
| 3 | 0.00288 +- 0.00042 | 0.00324 +- 0.00077 | 0.888 +- 0.014 | 405342 | 203 | 0.327 |
| 4 | 0.00190 +- 0.00032 | 0.00320 +- 0.00091 | 0.881 +- 0.031 | 534007 | 203 | 0.321 |

## edge_gate

| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |
|---|---|---|---|---|---|---|
| True | 0.00288 +- 0.00042 | 0.00324 +- 0.00077 | 0.888 +- 0.014 | 405342 | 203 | 0.327 |
| False | 0.05658 +- 0.01636 | 0.01398 +- 0.00085 | 0.078 +- 0.130 | 397818 | 203 | 0.000 |

## How to read this

- `mesh=none` is the same network with every edge deleted: an MLP.
- `jitter`/`random` keep node count and degree but break the honeycomb.
- `harmonics=0` removes the angular symmetry bias, `edge_gate=False`
  removes topology adaptation, `depth=1` removes the pyramid.
