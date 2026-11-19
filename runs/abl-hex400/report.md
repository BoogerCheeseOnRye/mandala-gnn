# Ablations

2 seeds x 400 steps, drawing `drawings/hex-tessellation.jpg`, mean +- sd over seeds.

`val` = held-out cells, `wedge` = one 45 deg sector never trained on. `mean-patch` is the do-nothing predictor: val 0.0224, wedge 0.0282.

## mesh

| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |
|---|---|---|---|---|---|---|
| hex_near | 0.00375 +- 0.00016 | 0.00305 +- 0.00052 | 0.893 +- 0.005 | 405342 | 203 | 0.270 |
| hex | 0.00184 +- 0.00005 | 0.00280 +- 0.00017 | 0.896 +- 0.010 | 405342 | 203 | 0.285 |
| rewired | 0.00223 +- 0.00112 | 0.00325 +- 0.00005 | 0.878 +- 0.001 | 405342 | 203 | 0.259 |
| jitter | 0.00422 +- 0.00095 | 0.00323 +- 0.00122 | 0.871 +- 0.020 | 405342 | 203 | 0.291 |
| random | 0.00362 +- 0.00011 | 0.00359 +- 0.00212 | 0.882 +- 0.038 | 405342 | 203 | 0.273 |
| grid | 0.00522 +- 0.00013 | 0.00235 +- 0.00022 | 0.960 +- 0.002 | 405342 | 196 | 0.321 |
| rings | 0.00285 +- 0.00023 | 0.00089 +- 0.00018 | 0.308 +- 0.080 | 405342 | 61 | 0.242 |
| chart | 0.00130 +- 0.00029 | 0.00065 +- 0.00002 | 0.520 +- 0.138 | 405342 | 25 | 0.248 |
| none | 0.01745 +- 0.00662 | 0.01770 +- 0.00434 | 0.730 +- 0.002 | 405342 | 196 | 0.000 |

## How to read this

- `mesh=none` is the same network with every edge deleted: an MLP.
- `jitter`/`random` keep node count and degree but break the honeycomb.
- `harmonics=0` removes the angular symmetry bias, `edge_gate=False`
  removes topology adaptation, `depth=1` removes the pyramid.
