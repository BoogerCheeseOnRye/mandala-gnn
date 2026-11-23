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
| `B(edge) = MLP(RBF(r), cos/sin kθ, radii, type)` | yes | relative-position bias on token pairs, whenever you can define a geometry for a pair |
| learned edge gate `g_ij ∈ (0,1)` | yes | soft sparsity / learned attention mask; readable as "which relations mattered" |
| `ChaosPyramid` gated residual pooling | yes | multi-resolution attention; already close to standard practice |
| 6,946-param energy head | only as a ranker | needs pairs of completions of the same mask ordered by true error |
| the fixed 203/61/25-node lattice as *the model* | no | it is the experiment, not the architecture |
| per-drawing node counts, wedge masks | no | benchmark geometry, not a prior |
| the 1.00 classification number | as a sanity check only | 3 classes, 12 test graphs; it says the encoder reads arrangement, not that it generalises |

## The ladder

Each rung is a question with a number attached. R0 and R2 are done, both
negative and for different reasons (a sequence prior that has nothing to act
on; a topology that is not a better router). R1 is narrowed to code structure;
R3 needs a GPU; R4 needs R3.

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

### R1 — does the bias mechanism transfer to code structure? (a day)

**Question:** when the relation is known from the data (indentation, bracket
nesting, line distance, AST distance) rather than hypothesised from a drawing,
does a pair bias beat RoPE on a byte-level code model at matched parameters?

**Why narrowed:** R0 killed the drawing-as-positional-code idea, so R1 no
longer tests that. What survives is the *mechanism* — a learned RBF/harmonic
pair bias with a gate — applied to structure the corpus actually has. Text is
dropped; code is where the structure is real and where the user wants to end up.

**Corpora:** a pinned 2 MB Python subset, byte-level, so no tokenizer is in the
comparison. Report bits/byte, not loss.

**Method:** same harness, 5 epochs, 2 seeds. Arms: RoPE, absolute position,
pair bias on raw distance, pair bias on structural features. A shuffled-feature
control, as in R0.

**Cost:** 6–8 h of CPU.
**Pass:** pair bias within noise of RoPE, or better, on held-out files.
**Kill:** >2% worse. Then the mechanism is not a text prior either, and the
repo's contribution to language modelling is the negative result plus the
query-time story.

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

**Recipe, not a run:** 50–300 M hybrid transformer on code + text, RoPE plus an
optional geometric bias, AR objective, then an energy *reranker* trained on
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

## Energy's job from here

Not a landscape, not a loss. A **selector**: given K sampled completions of the
same mask, pick the one with the lowest true error. The current head cannot do
that because it was trained to rank *how much was seen*, not *how right the
answer is*. The next honest energy experiment is 200 pairs of completions of
the same mask, labelled by true error, and a held-out reranking accuracy
against random choice. Until that runs, the energy page says what it says.

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
* R2 had shown a topology gain → that would have been the paper, with the
  energy head as a side result. It did not: random wiring at matched degree is
  at least as good, so that route is closed rather than pending.
* What is left is the measured negative result plus the reusable layers. The
  next project starts from the layers, not from the drawings.
