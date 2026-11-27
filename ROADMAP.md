# Roadmap: what actually carries forward

Written after the three-drawing energy run (`73e7e5b`) and the Grok planning
thread. This is an internal plan, not a promise: every rung below has a kill
criterion, because the last two experiments were worth more when they failed
cleanly than they would have been as demos.

## Where the repo stands

| what | measurement |
|---|---|
| backbone | 405,342 params, 128×128 drawings, 64-d states |
| graphs | 203 / 61 / 25 nodes (hex, rings, chart) |
| masked reconstruction, wedge r | 0.9057 / 0.7663 / 0.9357 |
| classification, unseen rotations | 1.00 (3 drawings × 8 rotations, 4 held out) |
| energy head | closed as a landscape; open as a selector (24/24, 19/24, 14/24) |
| R0, geometry as a sequence prior | negative: order 4.534 bits/code, geometry 4.661, shuffled control 4.681 |
| R1, pair bias on code | negative: 2.555 bits/byte vs 2.481 for RoPE (+3.0%); structure beats distance-only and shuffled controls |
| R2, drawing as a computation graph | negative: drawing 0.996/1.000/1.000 AUC, matched random 1.000 everywhere |
| cost | ~80 s per 400-step run on 8 CPU cores, 0.25 s inference |

The two things that actually replicate are the **symmetry-aware radial
attention** (RBF distance + angular harmonics + learned edge gate) and the
**pyramid**. The hand-drawn lattice itself did not earn its keep at this task
scale, and R2 (below) tested the strongest remaining version of that claim — the
lattice as a computation graph — and it did not hold either. Any plan that
assumes otherwise is starting from the wrong result.

## The distance to "chat and code", in hours

The machine is an 8-core ARM CPU with 7.4 GB of RAM, no GPU. Measured
throughput for a causal transformer at dim 96, 3 layers, context 128, batch 8:
**217 ms/step, 4,721 tokens/s**. That is the budget. Chinchilla puts a
compute-optimal run at roughly 20 tokens per parameter, so:

| model | tokens | time on this device |
|---|---|---|
| 0.17 M (the bench above) | 3.4 M | ~12 min |
| 1 M | 20 M | ~1.5 h |
| 10 M | 200 M | ~3–5 days |
| 100 M | 2 B | ~4 months |
| frontier chat/code (10⁹–10¹²) | 10¹²+ | not computable here |

So the on-device ceiling is ~1–10 M parameters on a few hundred MB of text or
code. That is enough for a real experiment and nowhere near a useful assistant.
Anything that says "train a 50–300 M model" in this repo is either a GPU
recipe or fiction; this plan treats it as a recipe (R3), not a rung we run.

Two consequences worth stating before the ladder:

* **A fixed lattice cannot become a language model.** Language has no 2-D
  carrier of 25–203 nodes. The drawing can be a *prior over positions*, a
  *computation graph*, or a *bias on derived structure* — those are the three
  bridges below, in increasing faithfulness.
* **Next-token prediction stays the main objective.** Energy-based language
  models are a research frontier, not a drop-in replacement; the energy head's
  measured behaviour (flat along the quality axis) is a reason to use it as a
  ranker, not as the loss.

## What transfers, and what does not

| piece | transfers to sequences? | how |
|---|---|---|
| `B(edge) = MLP(RBF(r), cos/sin kθ, radii, type)` | yes, but measured worse | relative-position bias on token pairs; R1 put it 3% behind RoPE on code |
| learned edge gate `g_ij ∈ (0,1)` | yes | soft sparsity / learned attention mask; readable as "which relations mattered" |
| `ChaosPyramid` gated residual pooling | yes | multi-resolution attention; already close to standard practice |
| 6,946-param energy head | only as a ranker | needs pairs of completions of the same mask ordered by true error |
| the fixed 203/61/25-node lattice as *the model* | no | it is the experiment, not the architecture |
| per-drawing node counts, wedge masks | no | benchmark geometry, not a prior |
| the 1.00 classification number | as a sanity check only | 3 classes, 12 test graphs; it says the encoder reads arrangement, not that it generalises |

## The ladder

Each rung is a question with a number attached. R0, R1 and R2 are done, all
negative and for different reasons: a sequence prior with nothing to act on, a
positional mechanism that loses to RoPE, and a topology that is not a better
router. R3 needs a GPU; R4 needs R3.

### R0 — does the bias work where the geometry is real? (hours)

**Question:** on next-patch prediction over the drawings themselves, does a
radial geometry bias beat a matched absolute-position transformer, and does it
survive rotations the absolute model has never seen?

**Why first:** if the bias does not help when the data *is* the geometry,
transplanting it to text is a waste of a week. This is the control the energy
work taught us to run.

**Method:** synthetic renders of the three lattices (`mandalanet/synth.py`):
600 train and 75 held-out drawings per layout, each with fresh node jitter,
rotation, scale, stroke width and ink, so held-out samples are new drawings
rather than new crops of three photographs. A 128-entry patch codebook gives
the sequence, visited in scanline order; the model (`mandalanet/seq.py`) is a
causal transformer, dim 80, 3 layers, 4 heads, matched across arms. Six arms:
`none`, `abs1d` (raster index), `abs2d` (Fourier features of absolute x,y),
`geom` (RBF distance + angular harmonics), `geom_rbf` (distance only, the
rotation-invariant half), `geom_shuf` (same machinery, shuffled coordinates, a
capacity control). Metric: bits per patch code.

**Cost:** ~1 h for 6 arms × 2 seeds on 8 CPU cores; data build ~3 min, cached.
**Pass:** `geom` or `geom_rbf` beats `abs2d` by >2% bits/code on held-out
renders, and `geom_shuf` stays near `none`.
**Kill:** no gain. Then drop bridges 1–2 and go straight to R2, where topology
is the computation rather than a position code.

**Result (6 arms × 2 seeds, 60 min).**

| arm | bits/code | vs order |
|---|---|---|
| raster index (`abs1d`) | **4.534** | — |
| absolute (x,y) (`abs2d`) | 4.558 | +0.023 |
| distance + angle (`geom`) | 4.661 | +0.126 |
| distance only (`geom_rbf`) | 4.669 | +0.135 |
| shuffled control (`geom_shuf`) | 4.681 | +0.147 |
| no position (`none`) | 4.700 | +0.166 |

Order beats every geometry arm by ≥0.12 bits, and the true geometry sits within
0.03 bits of its shuffled control. Per layout, absolute position wins only on
the 203-node hex drawing (5.812 vs 6.002); order wins on rings (4.065 vs 4.300)
and chart (3.537 vs 3.803). Seed spread is ≤0.011 bits, so the ordering is not
noise.

**Killed as stated, for a structural reason.** A causal next-token model is
asked for token t+1, whose position is already fixed by the order; there is no
query position for geometry to route to. Geometry is a *query-time* prior —
masked reconstruction already showed it helps there — not a sequence prior.
That kills bridge 1 (a drawing as a positional code) outright, and makes bridge
3 (structure known from the data, e.g. an AST) the only remaining text route.
R2 is unaffected and becomes the main line.

### R1 — does the bias mechanism transfer to code structure? (done, negative)

**Question:** when the relation is known from the data (indentation, bracket
depth, line distance) rather than hypothesised from a drawing, does a pair bias
beat RoPE on a byte-level code model at matched parameters?

**Why narrowed:** R0 killed the drawing-as-positional-code idea, so R1 no
longer tests that. What survives is the *mechanism* — a learned RBF pair bias,
here without the angular harmonics because code has no angle — applied to
structure the corpus actually has. Text is dropped; code is where the structure
is real.

**Corpus:** a pinned subset of CPython 3.12.0's `Lib/` (tarball
`codeload.github.com/python/cpython/tar.gz/refs/tags/v3.12.0`, extracted from
`Lib/`), byte-level, 1.85 MB train / 169 KB held out, split by *file* so no
held-out byte comes from a file the model saw. The corpus sha256 and the exact
file list are in `runs/code/results.json`.

**Method** (`mandalanet/code.py`, `tools/code_experiment.py`): dim 96, 3 layers,
4 heads, context 256, batch 4, 1200 steps (~1.2 M tokens, about 0.7 epochs),
dropout 0.1, two seeds. Five arms that differ only in how attention learns
position:

| arm | position source |
|---|---|
| `rope` | rotary position on q/k — the baseline to beat |
| `abs` | learned absolute position embedding |
| `bias_dist` | learned bias from RBF(|i-j|); no RoPE, no abs |
| `bias_struct` | bias from RBF(|i-j|) + indent, bracket depth, line distance, same-line flag; no RoPE, no abs |
| `bias_shuf` | the same features with the structural channels permuted across positions — capacity control |

In the three bias arms the bias is the *only* source of position, so this asks
whether the mechanism can replace RoPE, not whether it can decorate it. Metric:
bits/byte on held-out files.

**Cost:** ~2.5 h of CPU for 5 arms × 2 seeds.
**Pass:** a bias arm within noise of RoPE, or better.
**Kill:** >2% worse than RoPE. Note that "within noise" is not a win: it means
the mechanism is harmless, not that it is a code prior.

**Result** — held-out bits/byte, two seeds, at two step budgets:

| arm | 300 steps | vs RoPE | 1200 steps | vs RoPE |
|---|---|---|---|---|
| RoPE | **3.229** | — | **2.481** | — |
| bias: distance + structure | 3.370 | +4.4% | 2.555 | +3.0% |
| bias: distance only | 3.423 | +6.0% | 2.612 | +5.3% |
| bias: shuffled (control) | 3.441 | +6.6% | 2.651 | +6.8% |
| absolute position | 3.613 | +11.9% | 3.345 | +34.8% |

Seed spread is ≤0.006 at 300 steps and ≤0.010 at 1200, so every gap above is
7–20× noise, and the ordering is identical at both budgets.

**Verdict: killed as stated.** The structural bias is 3.0% worse than RoPE at
the full budget — over the 2% kill line — and it had the advantage of being the
only positional signal rather than competing with one. It is also slower to
learn: the gap is *larger* at 300 steps (+4.4%) than at 1200 (+3.0%).

Two things in the run are worth keeping even though the rung fails.

* **The structural features carry real information.** Inside the bias family,
  `distance + structure` beats `distance only` by 2.2% and the shuffled control
  by 3.6% at 1200 steps (1.5% and 2.1% at 300). The shuffled control has the
  same capacity and the same feature distribution with the channels permuted
  across positions, so that gap is the information, not the machinery.
  Indentation, bracket depth and line distance are learnable relations; they
  are simply not worth 3% of bits/byte against RoPE.
* **Absolute positions collapse** (+34.8%). With random windows, a small model
  and byte-level code, learned offsets memorise rather than generalise. That is
  the clearest argument in the whole repo *for* relative mechanisms — just not
  for this one.

**Caveat, stated plainly:** at 1200 steps the model is overfitting (train 1.13
nats against 1.82 nats held out), so both columns are past-the-optimum
checkpoints. They are matched across arms and the ranking does not move between
them, but they are not each arm's best number.

### R2 — is the drawing a better computation graph than a random one? (done, negative)

**Question:** put a recurrent net's state on the drawing's nodes, let messages
follow its edges for k steps, and ask whether a target node is within k hops of
a source node. Does the drawing compute as well as a *connected* random graph
with exactly the same node and edge counts, and better than a chain or a star?

**Why it matters:** this is the project's actual claim — topology from the
drawings as layers — and it does not require the lattice to be a language
model. It also uses all three drawings at once, which the reconstruction task
never really did.

**Method.** `mandalanet/compute.py` is a GRU cell with weights shared across
steps, mean-aggregated messages over a fixed edge set, and no node embeddings.
`tools/reach_task.py` runs the probe. k is chosen *per graph* as the hop budget
whose positive rate is closest to 50%, and the model gets exactly k steps, so
no substrate is asked something its own wiring cannot answer. Pairs are sampled
half positive, half negative; 20% of source nodes are never trained on. The
primary metric is AUC, which is threshold-free and independent of the class
mix. Substrates: the drawing (edges symmetrised — chart and rings ship one-way
edges), a connected random graph with identical node and edge counts, a
double-edge-swapped copy of the drawing (identical degrees, no geometry), a
chain and a star. 500 steps, dim 48, batch 32, two seeds (rewired: one).

The first version of this task regressed the BFS distance to every node
(`tools/topology_task.py`, kept in `runs/topology/results.json` as the pilot).
It was not a fair comparison: the drawings' diameter is 6, so the answer sat in
a handful of classes whose mix differs per graph, and balanced accuracy mostly
measured those priors. Chart saturated for every substrate, rings favoured
random, hex was under-trained at 500 steps. The binary probe replaced it.

**Result** (AUC and balanced accuracy per seed; k is each graph's own hop
budget, diam its diameter):

| layout | graph | k | diam | edges | AUC (per seed) | balanced acc |
|---|---|---|---|---|---|---|
| chart | drawing | 2 | 6 | 64 | 1.000/1.000 | 1.000/1.000 |
| chart | random | 2 | 4 | 64 | 1.000/1.000 | 1.000/1.000 |
| chart | rewired | 2 | 4 | 64 | 1.000 | 1.000 |
| chart | chain | 7 | 24 | 24 | 1.000/0.693 | 1.000/0.608 |
| chart | star | 1 | 2 | 24 | 1.000/1.000 | 1.000/1.000 |
| rings | drawing | 3 | 6 | 186 | 1.000/1.000 | 1.000/1.000 |
| rings | random | 2 | 4 | 186 | 1.000/1.000 | 1.000/1.000 |
| rings | rewired | 2 | 4 | 186 | 1.000 | 1.000 |
| rings | chain | 16 | 60 | 60 | 0.325/0.673 | 0.542/0.567 |
| rings | star | 1 | 2 | 60 | 1.000/1.000 | 1.000/1.000 |
| hex | drawing | 4 | 12 | 1046 | 0.996/0.982 | 0.982/0.951 |
| hex | random | 2 | 4 | 1046 | 1.000/1.000 | 1.000/1.000 |
| hex | rewired | 2 | 4 | 1046 | 1.000 | 1.000 |
| hex | chain | 16 | 202 | 202 | 0.573/0.593 | 0.545/0.573 |
| hex | star | 1 | 2 | 202 | 1.000/1.000 | 0.928/0.928 |

**Verdict: killed as stated.** Nothing beats the matched random graph. The
drawing ties it on chart and rings (both 1.000) and trails it slightly on hex
(AUC 0.996/0.982 vs 1.000/1.000, balanced 0.982/0.951 vs 1.000). It also
converges later: at step 166, rings drawing 0.945/0.573 against random
1.000/1.000; hex 0.924/0.835 against 1.000/1.000.

The rewired control is the sharp part. It has the drawing's degree sequence and
edge count exactly, random wiring, and it is indistinguishable from the random
graph (1.000 everywhere, diameter 4). So the drawing's deficit is not degree
heterogeneity interacting with mean aggregation; it is its *specific* wiring.
At fixed nodes and edges a mesh is a longer graph (diameter 12 vs 4 on hex) and
needs more hops for the same question (k=4 vs k=2). Random wiring at fixed
degree is an expander; a lattice is not. On a reachability question that is the
whole difference, and it favours random.

Chains are the only outright failures, and they fail measurably: on rings and
hex the training loss sits at exactly ln 2 = 0.6931 at one or more checkpoints —
the output never leaves the constant predictor — and AUC swings 0.325–0.673
across seeds. A 16-hop signal under mean aggregation on a degree-2 path is
attenuated by about 2^-16, so there is no gradient to learn from. That is a
property of a path, not evidence for the drawing, and it makes a chain the
wrong control for this task. Stars pass trivially because k=1 asks "is it a
direct neighbour", which hub/leaf structure answers in one step.

**Consequence.** Topology is not the ingredient. The drawing is *sufficient*
(1.000 on two of three layouts) but not *superior*, and where it differs it is
worse. R3 and R4 are therefore built on the mechanism that did replicate — the
RBF/harmonic pair bias, the learned edge gate, the pyramid — and not on the
edges. This agrees with the reconstruction ablation (rewired ≈ hex) and with R0
(geometry inert as a sequence prior).

**What this does not show.** Not that geometry never helps. It shows that on a
reachability question — where random expanders are provably good routers — the
drawing is not a better place to compute. A task defined on the drawing's own
coordinates (same sector, same ring) would favour it by construction and would
not be evidence. A task whose *structure* matches the drawing's symmetry without
being its coordinates is the untested middle; nothing in these results suggests
it would pay for the effort.

### R3 — scale, if compute appears (GPU, weeks)

**Recipe, not a run:** 50–300 M hybrid transformer on code + text, RoPE as the
position mechanism (R1 measured the pair bias 3% worse, so it is not in the
default recipe), AR objective, then an energy *reranker* trained on
quality-ordered pairs (same mask, different completions, ordered by true
error). Evaluate bits/byte, HumanEval, MBPP. The energy component is allowed to
help only where it is measured to help: constrained decoding, reranking, or
refinement — never as the only objective.

**Deliverable here:** config, data mix, eval scripts, and a baseline table from
R1 so the scaled run has something honest to beat.

### R4 — the model proposes its own layers (after R3)

"Models writing layers" is architecture search. It needs a proposer that is
already a decent coder and a verifier that is cheaper than the search. A micro
version is possible here: let a proposer emit variants of the bias (harmonics,
edge set, gate shape) and score them on the R0/R2 harness. That is a useful
tool; it is not self-improvement, and it should not be described as such.

## Energy's last job: the selector, measured

Not a landscape, not a loss. A **selector**: given K completions, pick the one
with the lowest true error. That was the last open claim, and it is measured
now in four candidate regimes, on all three drawings (`tools/energy_selector.py`,
`runs/energy-selector/results.json`, about 20 minutes of CPU).

**Protocol.** 60 training masks and 40 held-out masks plus the wedge, per
drawing, at ratios 0.15–0.35 of the visible cells — none of them in the
visibility bank the old head was trained on. K = 8 candidates per mask. Labels
are the masked MSE of the decoded patches, so they never touch the head's
input. Three heads are scored on identical candidates: the published
visibility head, a head trained with a pairwise hinge on the true ordering, and
the same head trained on the *opposite* ordering. The inverted head is the
control that matters: it has to land below chance wherever the labels are
learnable, or the loop is not learning anything.

Held-out pairwise accuracy, chance 0.5 (`quality` head unless marked):

| candidates | what differs | hex | mandala | chart | error spread |
|---|---|---|---|---|---|
| same mask, fixed temperature (σ 1.0 / 2.0) | noise realisation only | 0.463 / 0.475 | 0.447 / 0.553 | 0.514 / 0.539 | 20–37 / 13–28 / 20–48% |
| **same input, K dropout samples** | the model's own noise | **0.500** | **0.505** | **0.507** | 40 / 8 / 76% |
| same target mask, same number visible | which cells were seen | 0.671 | 0.506 | 0.681 | 41 / 5 / 3% |
| same target mask, different amounts visible | how many cells were seen | 0.807 | 0.649 | 0.562 | 46 / 5 / 2% |
| *neighbour smoothness*, swap family (no learning) | — | 0.628 | 0.499 | 0.603 | — |
| *count of visible cells* (no learning) | — | 0.794 | 0.642 | 0.525 | — |
| *inverted labels*, same-mask sampler | — | 0.473 | 0.461 | 0.462 | — |

**Verdict: closed as a reranker, with one capability worth naming.**

* **On the case a reranker actually faces — K samples of the same input — the
  head is at chance on all three drawings**: 0.500, 0.505, 0.507, against
  training-mask scores of 0.598–0.613. Pooled over drawings the sampler family
  is 0.504 ± 0.014 (123 mask-cells) and the fixed-temperature family is
  0.498 ± 0.009 (246): at chance, with the interval excluding any useful
  reranking. So it learns something that does not transfer, every time, at
  every temperature. A neighbour-smoothness statistic
  with no parameters and no training beats it on the hex drawing (0.598, ahead
  by 0.097, paired t = 2.7) and the chart (0.590, ahead by 0.083, t = 2.2); it
  loses on the mandala (0.451).
* **Where it does order above chance, it is the setting a reranker never has.**
  Given candidates that each hid a different random subset of the *same number*
  of cells, it reaches 0.671 on hex and 0.681 on the chart, and on hex its pick
  is 5.6% better than an average candidate (37% of the available improvement).
  That is "which observation was better", not "which sample is better": you do
  not sample different subsets of your own input and ask the head to choose.
* **Where the candidates differ in how much they saw, it reproduces a count.**
  0.807 against the count's 0.794 on hex (+0.014 ± 0.005), 0.649/0.642 on the
  mandala, 0.562/0.525 on the chart. Nothing a head gives you that counting
  visible cells does not.
* **The mandala is uninformative in two of the four families**: its candidates
  differ by 5–8% in error, so there is almost nothing to rank there. That is a
  property of the drawing, not of the head, and it is why every claim here is
  reported per drawing.

The earlier visibility result (24/24, 19/24, 14/24) was underpowered on the
small graphs rather than wrong: with 40 masks × 28 pairs the visibility-trained
head ranks by amount-seen at 0.781/0.556/0.506. It still adds nothing over the
count.

So the search loop stays closed, and the reason is now precise: the head's
signal is about *how much was seen*, which you already know, and when it is
made to rank quality at fixed input it memorises masks instead. The one thing
it can do that a count cannot is rank which subset of a fixed-size observation
was more informative — and that is not the problem a search loop has.

## Images and code generation: ten steps, small and fast

Two artifacts exist and both are small: the 405 k GNN reconstructs the three
drawings at r 0.906 / 0.742 / 0.936, and the 385 k byte-level transformer writes
Python at 2.48 bits/byte on files it never saw. The target is one model that
does both, under a stated envelope: **≤ 5 M parameters, ≤ 25 MB int8, ≥ 1 kB/s
generation on this phone, ≤ 2 s to finish a drawing**. Every step below has a
gate and a kill, and every number comes from `tools/` on this device.

1. **The meter (done, `runs/budget/results.json`).** One script reports params,
   MB, bytes/s, held-out bits/byte and compile rate for code, and per-pass cost
   plus a progressive-fill image, for the drawing model. References: the R1
   RoPE byte model retrained and saved, an order-2 byte n-gram floor, and the
   hex GNN's eight-pass fill against a constant-image floor. Later steps are
   scored only against this table. *Gate: the table exists and is reproducible
   with one command.*
2. **Evaluation sets of record (done, `runs/budget/eval.json`).** 40
   hand-written Python tasks (signature + docstring prompt, unit tests, and a
   reference body that has to pass those tests) and 36 synthetic drawings
   (12 per layout, seeds 1000–1011), frozen under one SHA. `make_eval.py`
   checks that no task prompt occurs in the training bytes and that the 12
   held-out files are actually held out; `--selftest` and `--verify` are in
   the smoke tests. *Gate: the sha verifies and every solution passes.*
3. **The code model under a fixed token budget (done, negative).** Two arms:
   1.26 M parameters at the baseline's 1.2 M tokens, and the 385 k model at 3×
   the tokens (3.7 M). Bits/byte falls on both — 2.478 → 2.296 and 2.156 — but
   no body compiles and no test passes. *Gate missed on both metrics.*
   *Kill triggered: the byte model is a completion model at this budget, and
   step 5 is cut.*
4. **Fast sampling (done).** A KV cache takes the 385 k model from 87 to
   737 B/s in torch; the same weights decoded in numpy reach 1,075 B/s on one
   stream (2,032 B/s over four) with logits matching torch to 6e-6 and the same
   greedy bytes on all 44 prompts. *Gate met: 1,075 B/s, +0.02% bits/byte,
   0.42 MB on disk.*
5. ~~Conditional code.~~ **Cut by step 3's kill, as pre-registered.** The
   40-task pass@1 stays in the eval set as the record of what the byte model
   cannot do; it is not retried at a slightly larger size without a new,
   separately pre-registered plan (see "What would change the plan").
6. **The drawing generator (done, gate missed by both).** Two candidates on the
   same held-out drawings: the GNN's masked-fill loop (8 and 16 passes), and an
   AR patch-token decoder at matched parameters. Score masked MSE, r, and
   seconds per drawing against the constant-image floor. *Gate: beats the floor
   by 2× MSE and finishes in ≤ 2 s.* *Kill: both at the floor — then "images"
   means reconstruction, not generation, and the page says so.* Result: 0.55×
   the floor (60% of the representable gain) at 0.48 s for the fill loop, 0.81×
   (25%) at 0.36 s for the decoder. The gate is missed; the kill's condition is
   not literally met, so the page says that instead.
7. **One trunk, two alphabets.** A single decoder-only transformer with a byte
   embedding and a patch embedding, geometry bias only in the patch attention
   (R0's lesson: geometry is a query prior). With step 5 cut, the code side is
   scored by bits/byte, not pass@1. Matched-parameter control: the same code
   model and image model trained separately, same total parameters and tokens.
   *Gate: the shared trunk is within 3% bits/byte and 10% image MSE of the
   two-model control; otherwise the sharing is not free and gets dropped.*
8. **Transfer, pre-registered.** Image-pretrain then code-fine-tune, and the
   reverse, at fixed tokens and parameters, two seeds. Bar: ≥ 2% bits/byte or
   ≥ 5 points compile rate. *Kill: both transfers inside seed noise — publish
   the null and stop selling "multimodal".*
9. **Conditioning without a chat stack.** Twelve fixed commands, completion
   style rather than docstring-to-function, split across the two skills ("draw
   the hub chart", "finish this line: total = sum(...)"), a documented prompt
   format, and success measured per command. *Gate: ≥ 8/12 correct.* *Kill:
   ≤ 4/12 — ship completion-only.*
10. **Ship and decide.** int8 weights ≤ 25 MB, on-device timing, a page with
    samples, the meter table, and the failures. If steps 5 and 6 stay at their
    floors, keep the meter and the negative result, and stop — the point of
    the ten steps is to make that decision with numbers, not to keep the
    branch alive.

The cheapest useful thing to build first is the meter: every later claim is a
delta against it, and it costs one training run and one afternoon.

### Step 1, measured

`tools/gen_meter.py --out runs/budget` (one command; checkpoint saved to
`runs/budget/code-rope.pt`, samples in `runs/budget/samples.txt`).

| what | code | image (hex) |
|---|---|---|
| parameters / fp32 size | 385,120 / 1.54 MB | 405,342 / 1.62 MB |
| held-out quality | **2.478 bits/byte** (R1's 2.478, reproduced) | wedge r **0.906** at 130 ms (published 0.906) |
| generation speed, 1 thread | **65–83 B/s** (96 bytes, 4 prompts) | **133 ms** per forward |
| generated output | **1 of 16 samples compiles** (6.2%); order-2 byte floor 6.211 b/B, 0 of 12 compile | 8-pass fill from nothing: **1.0 s**, r **0.360** |
| floor | n-gram, no learning | constant image: r 0.287, MSE 0.169 |

Three things the meter already says, before any new model:

* **The code model is real but slow.** 2.478 bits/byte is the R1 number
  reproduced exactly, and a generated continuation compiles — but one sample
  in sixteen, and 65–83 B/s is 12–15× below the ≥ 1 kB/s gate. Both are fixable
  (step 4's KV cache; step 3's budget), and neither is fixed yet.
* **Greedy decoding is degenerate at this size.** The four greedy samples run
  into repeated spaces and `self._code(self, self.____`; sampled output at
  temperature 0.8 is the one that compiles. Later steps report greedy and
  sampled separately, because averaging them hides that.
* **The drawing model cannot generate from nothing.** Its eight-pass fill
  lands at r 0.360, against 0.287 for a constant image and 0.906 for the same
  checkpoint's wedge completion — about a tenth of the way from a blank image
  to its own conditional answer. The checkpoint is a reconstructor; a
  generation claim has to be earned with a schedule it was trained for, or
  with a different decoder (step 6). This is the most important row in the
  table.


### Step 2, measured

`tools/make_eval.py` → `runs/budget/eval.json`, sha **1913f265…**, rebuilt and
re-verified (40 tasks, sha `956313d5…`; 36 drawings, sha `226e2557…`; corpus
sha matches the training data). All 40 reference solutions pass their own
tests, all 36 drawings re-render to their hashes, and no task prompt or
held-out path occurs in the training bytes.

The step-1 model, scored on that set:

| | greedy | sample t=0.8 |
|---|---|---|
| naive `compile()` | **40 / 40** | 4 / 40 |
| compiles with a real body | **0 / 40** | 0 / 40 |
| passes the task's tests (pass@1) | **0 / 40** | 0 / 40 |

The first row is the one to keep: greedy decoding emits 96 blank characters
after the docstring, which is valid Python, so a naive compile check reports
100% and means nothing. The scored metric requires a statement after the
docstring, and on that metric the model has written no function body at all.
Its one sampled success in step 1 was an in-distribution CPython continuation,
not an answer to a docstring.

That sets the floors honestly: step 3 has to get a *body* to compile on more
than four of these prompts, and step 5's 5/40 bar starts from zero, where any
test that passes is a real signal. The 36 drawings are for step 6 and have not
been used yet.

### Step 3, first arm: capacity

`tools/code_budget.py --dim 128 --depth 6 --steps 1200` → 1,255,680 parameters
at the baseline's 1.23 M tokens, checkpoints at 600 and 1,200 steps.

| step | bits/byte | body compiles (greedy / sampled) | pass@1 |
|---|---|---|---|
| 600 | 2.506 | 0 / 40 · 0 / 40 | 0 / 40 |
| 1,200 | **2.296** | 0 / 40 · 0 / 40 | 0 / 40 |
| *385 k at 1,200* | *2.478* | *0 / 40 · 0 / 40* | *0 / 40* |

Capacity buys bits/byte (2.478 → 2.296, 7% lower) and buys nothing on the task
set: still no function body, still no test passed. One sampled completion did
parse as a body — and it was a second docstring, which the metric had counted.
The rule is now "a statement that is not a string literal", every checkpoint
was rescored from its saved completions (the text is unchanged), and the row
above is the corrected one. That is a one-line tightening of the metric, not a
change to any number that was measured.

Per the pre-registered rule, tokens are next: the 385 k model at 3,600 steps.

### Step 3, second arm and the decision

`tools/code_budget.py --dim 96 --depth 3 --steps 3600` → 385,120 parameters,
3,686,400 tokens (two epochs of the corpus), checkpoints every 1,200 steps.

| step | tokens | bits/byte | body compiles (greedy / sampled) | pass@1 |
|---|---|---|---|---|
| 1,200 | 1.23 M | 2.409 | 0 / 40 · 0 / 40 | 0 / 40 |
| 2,400 | 2.46 M | 2.205 | 0 / 40 · 1 / 40 | 0 / 40 |
| 3,600 | 3.69 M | **2.156** | 0 / 40 · 0 / 40 | 0 / 40 |
| *arm 1 (1.26 M params, 1.23 M tokens)* | *1.23 M* | *2.296* | *0 / 40 · 0 / 40* | *0 / 40* |
| *baseline (385 k, 1.23 M tokens)* | *1.23 M* | *2.478* | *0 / 40 · 0 / 40* | *0 / 40* |

The 1,200-step row is not a reproduction check: the cosine schedule spans all
3,600 steps, so its learning rate at step 1,200 is higher than the baseline's.
The final number, 2.156, is 13% below the baseline, and arm 1's 2.296 is a
clean capacity-at-equal-tokens comparison.

Across both arms that is 200 greedy and 200 sampled completions. Exactly one
sampled completion (2,400 steps, `merge_sorted`) parsed as a function body:

```python
    # bicteon comport the the defails=None

    def __is__(self):
        return intext(tambort)
```

It references an undefined name and fails every test. So: **the model writes
text that scores better on held-out files and still writes no functions.**

Per the pre-registered kill, conditional code (step 5) is cut. What is closed
is narrow: *a 0.4-1.3 M-parameter byte model, trained on 1.8 MB of CPython for
two epochs, does not follow a docstring.* That does not say a code model needs
to be this small, or that fine-tuning on (docstring, function) pairs could
never work — only that at this envelope, with this corpus, the cheap version
is measured and empty.

Why the kill is right rather than impatient: bits/byte is still improving
(2.478 → 2.156) while the task metric is flat at zero, so more of the same
compute buys a better language model of Python and not a code assistant. The
obvious reopen is a single pre-registered ceiling run — 5 M parameters and the
full ~30 MB `Lib/` corpus, several hours of CPU — with the same 40 tasks and
the same rule. That is a decision to take deliberately, not by grinding the
current size.

### Step 4, measured

`tools/fast_sampling.py`, one thread, `OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1`, 96 new bytes per prompt:

| runtime | 1 stream | 4 streams |
|---|---|---|
| torch, full window per byte | 87 B/s | — |
| torch, KV cache, oneDNN on | — | 445 B/s |
| torch, KV cache | — | 737 B/s |
| numpy, KV cache, int8 weights | **1,075 B/s** | **2,032 B/s** (1,464 sampled) |

The floor on this device is dispatch, not arithmetic. A (1, 96) x (96, 288)
matmul costs 400-1,400 µs through oneDNN and ~24 µs with oneDNN disabled, and
one cached torch step makes ~230 aten calls. The numpy sampler makes ~230 numpy
calls at ~2.8 µs each, so the fix is fewer ops: each layer norm's affine is
folded into the matmul that follows it, rope is one 24x24 matmul per call
instead of six ufuncs, and the variance reduction is a gemv. That took the same
weights from 433 to 1,075 B/s. The BLAS thread count is not a detail — with the
default eight threads every tiny matmul pays a pool wake-up and the loop runs
at 754 B/s.

int8 dynamic quantization made torch *slower* (232 → 137 B/s): it adds unpacking
per call while the call overhead dominates. In numpy the weights are
dequantized once at load, so int8 there is a storage format, not a speed knob.

Equivalence, because a fast sampler that generates different bytes is worthless:
the fp32 export matches torch's logits to 5.7e-6; the int8 export shifts logits
by up to 0.08 but produces byte-identical greedy output on the four meter
prompts and on all 40 frozen tasks; held-out bits/byte moves 2.4777 → 2.4782
(+0.02%).

Gate: 1,075 >= 1,000 B/s single-thread, +0.02% <= 1%, 0.42 MB <= 25 MB. Met. A
96-byte completion takes 89 ms; four at once take 47 ms each. That is enough for
a demo and is not a chat model.

### Step 6, measured: both candidates missed the gate

Two candidates, the 36 frozen drawings (12 per layout), one renderer, one
scoring path. `tools/draw_gen.py` scores the GNN's fill loop,
`tools/ar_patches.py` scores the AR patch decoder, and both report two controls:

* **floor** — the pixel-wise mean of 64 *training* drawings: the standard
  "predict the average image" baseline.
* **ceiling** — the drawing's own patches pushed through the same renderer. No
  model that emits one 4x4 patch per node can beat this, because the ink
  *between* nodes is not in its output space. Without it, "beats the floor by
  2x" cannot be read: on the chart the ceiling is 0.17x the floor and 2x is
  easy, on hex it is 0.39x and 2x asks for 67% of everything available.

The useful number is therefore **gain captured**, (floor - mse) / (floor -
ceiling). MSE and r are on covered pixels only (33.2% of a synthetic canvas; the
renderer leaves the rest black, which is the renderer's hole, not the model's).

| layout | nodes | floor | ceiling | A: fill, 8 passes | gain | B: AR codes | gain |
|---|---|---|---|---|---|---|---|
| hex | 203 | 0.0467 | 0.0181 | **0.0350** (r 0.493) | 41% | 0.0444 (r 0.327) | 8% |
| rings | 61 | 0.0302 | 0.0107 | **0.0241** (r 0.476) | 32% | 0.0465 (r 0.306) | -83% |
| chart | 25 | 0.1119 | 0.0189 | **0.0457** (r 0.681) | 71% | 0.0622 (r 0.538) | 53% |
| all | 96 | 0.0629 | 0.0159 | **0.0349** (0.55x floor) | 60% | 0.0510 (0.81x) | 25% |

Seconds per drawing, one thread, idle device: A 0.48 s (16 passes 0.96 s), B
0.36 s. Both pass the time bar; neither passes 0.50x floor. The gate is missed
by both, and the pre-registered kill's literal condition ("both at the floor")
is *not* met: A is 1.8x better than the floor and captures 60% of what the
representation allows, so "images means reconstruction, not generation" would
overstate the negative too. What is closed is the claim that either candidate
*is* a drawing generator at this size.

**Three explanations for A's miss, each tested and each refuted.**

1. *It was trained on photographs.* The photo-trained checkpoints score 0.0885
   on the same drawings (1.41x floor, r -0.047, gain -54%): they do not
   transfer at all, and training on synthetic drawings is what moves 1.41x to
   0.55x. Necessary, not sufficient.
2. *The fill loop's first pass is out of distribution.* `synth_gnn.py` sampled
   mask ratios 0.1-0.9, while a fill from empty starts at 1.0. Retrained with
   masks up to 1.0 (`runs/synth-gnn-empty`, `--mask-max 0.9` reproduces the
   first run): 0.0350, 0.56x floor, 59% gain. Inside noise of 0.0349 — the
   mismatch was real and it did not matter.
3. *The loop conditions on its own predictions and was never trained to.*
   `--teacher` runs the same loop but reveals the true patch after every pass,
   keeping the model's answer for the cells being scored. Result: 0.0349, the
   same to four decimals, and the largest change in the model's own output is
   0.0093 on a 0-1 range. The model is a low-pass filter: feeding it its own
   0.118-mean-error predictions barely moves its answer, so drift is not the
   limiter. Per-cell accuracy is.

**Candidate B, the AR patch decoder**, matched at 372,513 parameters (dim 96,
depth 3, 8 epochs x 125 steps, 1.6 M patch codes): 4.891 bits/code at 15.4%
accuracy, versus R0's 4.534 for a raster-index arm trained on less — so it is
worse per code as well as worse per drawing. Greedy decoding gives 0.83x floor
(23% gain) against 0.81x (25%) sampled, so decoding is not the problem. Rings is
worse than the floor: 61 tokens give a causal model almost no context to
condition on. Its one win is speed, 0.36 s against A's 0.48 s.

**What is closed and what stays open.** Closed: "a 405 k GNN draws these
drawings", "an AR patch decoder is the better small generator", and "the fill
loop's own error compounding is why the numbers stop at 0.55x". Open, and what
step 7 inherits: the missing 40% of the gap is per-cell prediction quality at
405 k parameters and 1,500 steps, and step 7's gate is a *delta* against a
two-model control, so it can be scored on these numbers without pretending they
are good.

Two methodology notes that belong with the numbers. The scoring tool pins
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before torch loads, because the same
run takes 0.48 s at one OpenMP thread and 2.15 s at the default eight. And the
frozen drawings are checked for leakage against B's training set (nearest
training drawing, largest per-node patch difference, 0-1 scale): hex 0.51, rings
0.38, chart 0.40. `tools/ar_patches.py --resume` re-scores a saved model and
writes the same JSON.

## Non-goals, stated once

* No OGB/LRGB leaderboard runs. A 405 k model will not be competitive there,
  the datasets are gigabytes, and the claim under test is "topology from
  drawings", not "better GNN". Matched-parameter controls are stronger here.
* No 50–300 M training on this device; that is months of CPU.
* No pure energy-based chat model. AR plus a ranker is the defensible hybrid.
* No claim that the 1.00 classification result implies language ability.
* No public "towards SOTA" page. The site reports measurements; this file
  reports intentions.

## What would change the plan

* R0 or R1 shows a real sequence gain → push the bias into R3 and publish it.
  Neither did: R0 found no sequence prior, R1 found the bias 3% behind RoPE.
* R2 had shown a topology gain → that would have been the paper, with the
  energy head as a side result. It did not: random wiring at matched degree is
  at least as good, so that route is closed rather than pending.
* A code generator inside this envelope → only via the ceiling run (5 M
  parameters, the full `Lib/`, the same 40 tasks and rule). R0-R2 and step 3
  say a small byte model on 1.8 MB will not get there by more steps.
* What is left is the measured negative result plus the reusable layers. The
  next project starts from the layers, not from the drawings.
