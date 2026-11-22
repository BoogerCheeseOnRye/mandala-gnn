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
| cost | ~80 s per 400-step run on 8 CPU cores, 0.25 s inference |

The two things that actually replicate are the **symmetry-aware radial
attention** (RBF distance + angular harmonics + learned edge gate) and the
**pyramid**. The hand-drawn lattice itself did not earn its keep at this task
scale. Any plan that assumes otherwise is starting from the wrong result.

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

Each rung is a question with a number attached. R0 is done (negative, and the
reason is structural); R1 is narrowed to code structure; R2 is the real test of
the thesis and takes days; R3 needs a GPU; R4 needs R3.

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

### R2 — is the drawing a better computation graph than a generic one? (days)

**Question:** give a small model a multi-step task where the computation graph
is explicit (pointer chasing on a grid, multi-digit addition with carries).
Put its state on the nodes and let messages follow the drawing's edges. Does
hex/rings/chart beat a chain, a star, a fully connected graph, and a random
graph of the same size and degree?

**Why it matters:** this is the project's actual claim — *topology from the
drawings as layers* — and it does not require the lattice to be a language
model. It uses all three drawings at once, which the recon task never did.

**Cost:** 2–4 days of CPU for a sweep with 2 seeds.
**Pass:** drawing topology beats random and chain on held-out instances of two
different tasks, at matched parameters and steps.
**Kill:** no consistent margin. Then the drawings are an aesthetic and a
dataset, and the repo should say so.

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
* R2 shows a topology gain → that is the paper, and the energy head becomes a
  side result.
* All three fail → the repo's contribution is the measured negative result
  plus the reusable layers, and the next project starts from the layers, not
  from the drawings.
