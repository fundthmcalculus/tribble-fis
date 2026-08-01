# Findings index

All 31 findings, with where each is recorded and what it changed. Narrative in
`RESULTS.md`; raw chronology in `LOG.md`.

**Legend** — 🔴 refutes a claim I made · 🟢 supports a claim · 🔧 engineering/method ·
⚠️ a mistake I made and caught

| # | Finding | Type | Entry |
|---|---|---|---|
| 1 | Fused pooling: the sequence pool commutes inside the consequent, so the expert projection runs once per *document*, not per token. Cuts a 4 GB intermediate; makes `d_out` cheap. | 🔧 | E001 |
| 2 | **Two degeneracies, not one.** Softmax saturation (τ too low ⇒ hard router) *and* uniform blending (τ too high ⇒ single averaged expert). Both give one effective expert. Rule-usage entropy detects only the first. | 🟢 | E001 |
| 3 | The contrastive objective *prefers* a crisp router: calibrated 0.500 → trained 0.039. | 🔧 | E001 |
| 4 | Training bottleneck was `NO_DUPLICATES` batch sampler, not the model. Profiling ruled out tokenisation (61.7k texts/s) and compute (55.5k samples/s, 0.16 GB). | ⚠️🔧 | E002 |
| 5 | Harness validated: two NanoBEIR figures reproduce published values to 4 decimals. | 🔧 | E002 |
| 6 | The rule base has *worse* training loss but better retrieval — regularisation or undertrained experts? (Resolved by #23.) | 🔧 | E003 |
| 7 | Calibrated τ = 1.195 at **both** D=64 and D=128 — the required temperature does not vary with `D`, so HTSK's `1/D` normalises the wrong quantity. | 🔴 | E003 |
| 8 | **Half the apparent gain was parameters.** Against a matched control: aggregate −0.06, retrieval +0.0187 (vs +0.56 / +0.0379 unmatched). | 🔴 | E004 |
| 9 | **Every fuzzy ingredient is neutral or harmful.** 50× range of routing softness → flat quality. HTSK ≈ product t-norm because τ and σ are the same parameter. Retracted my "HTSK is mandatory" claim. | 🔴 | E004 |
| 10 | The rule base trades smooth geometry for discrimination: Retrieval +3.8, Clustering −2.5, STS −1.3. | 🟢 | E004 |
| 11 | Classification deficit vs potion hypothesised as data, not architecture. (Confirmed by #18.) | 🔧 | E004 |
| 12 | FES-C collapsed to `H_fire` = 0.000; hypothesised query–document symmetry break. | 🔴 | E004 |
| 13 | The MTEB-14 → published offset is **family-specific** (+1.25…+1.42 static, +4.42 transformer), so cross-family aggregate comparison is invalid. Narrowed my earlier "consistent offset" claim. | ⚠️ | E005 |
| 14 | **FES-C refuted, two mechanisms.** Hard routing breaks query/document comparability; soft routing gets *deliberately neutralised* (model paid ~2.3 loss to become context-free). | 🔴 | E006 |
| 15 | Low rank decouples `R` from *parameters* but not *compute* — the antecedent is `O(L·R·d_in)` and does not factor out of the pool. | 🔧 | E007 |
| 16 | **`d_out` is nearly free**: 3× width for 5% throughput and 3.5% parameters. | 🟢 | E007, E016 |
| 17 | The rule base costs **3.67× CPU throughput** for a retrieval-only gain. | 🔴 | E007, E016 |
| 18 | **The classification gap is data, not architecture.** A0 (no rule base) shows the same 5.7-point deficit. Free test from a rung queued for another purpose. | 🟢 | E010 |
| 19 | **The two scaling levers are complementary**: table width buys the classification family, the rule base buys retrieval. A4 at 2.52M beats A0 at 7.91M on Retrieval. | 🟢 | E010 |
| 20 | **Interpretability refuted.** Prototypes decode to nothing; `‖A_r‖` spread 3.1%; emergent-IDF correlation +0.080. The model vector-quantises a semantically unanchored space. | 🔴 | E008 |
| 21 | **Retrieval gain replicates with complete separation.** +0.0221 (+5.47%), every A4 run beats every A1b run, exact permutation p = 0.050 (the floor at n=3). | 🟢 | E010 |
| 22 | **Aggregate null is definitive**, not under-powered: −0.12, p = 0.750, seed spreads 0.16–0.30. | 🔴 | E010 |
| 23 | **Experts are not undertrained.** 4× and 15× the dense LR both fall below A4's three-seed minimum. Kills the alternative explanation for #6 and protects #21 from a "tuned differently" objection. | 🟢 | E011 |
| 24 | **Low-rank consequent caps the whole model's embedding rank** at `k + min(R, d_out)`, because `V` is shared across rules. My equivalence test verified the algebra and missed the consequence. | ⚠️🔴 | E012 |
| 25 | The rank-bottleneck diagnosis confirmed by the uncontaminated metric: NanoBEIR recovered +0.105 purely from removing rank. | 🟢 | E013 |
| 26 | **Retrieval tracks usable embedding rank monotonically**; predicted bound (`k+R`) verified by SVD to +1. | 🟢 | E014 |
| 27 | **Scaling ordering `d_out` > `d_in` > `R`**, from configs 0.3% apart in parameters. `R` saturates by 32–64; S5 spent 3.7M extra parameters for nothing. | 🟢 | E014, E015 |
| 28 | **An epoch is worth more than a parameter, for retrieval**: 2.52M × 2 epochs matches 5.00M × 1 epoch. Implies every result here is undertrained. | 🔧 | E014 |
| 29 | **LogTSK reproduces the uniform degeneracy**, exactly as #2 predicted: `H_fire` = 0.999, lands *below the R=1 control*. Prediction registered before the run. | 🟢 | E014 |
| 30 | **Complete R-sweep**: retrieval rises monotonically with R; MTEB-14 shows no trend. The sharpest statement of the central result. | 🟢 | E014 |
| 31 | **Fixed-seed runs are not bit-reproducible** — 0.0054 NanoBEIR of GPU kernel noise. Sets a measured threshold: differences below ~0.01 are noise. | 🔧 | E015 |

## Mistakes caught, and what caught them

Worth keeping separate, because the pattern is more useful than any single entry.

| Mistake | How it presented | What caught it |
|---|---|---|
| `NO_DUPLICATES` sampler stalling training | 1% GPU utilisation, looked like a model problem | Profiling the two candidate components before guessing |
| Concurrent training exhausting VRAM | 320 s/it on a model that profiles at 37 ms — no OOM error, WDDM spilled silently | The internal inconsistency between measured and observed speed |
| `pgrep` guard that never guarded | Log said "launched"; GPU sat at 0 MiB | Checking *is it computing?* rather than *did I launch it?* |
| `Get-Process \| Where CommandLine` killing nothing | No error; the process simply survived | `Get-Process` has no `CommandLine` property — `Get-CimInstance` does |
| Low-rank consequent capping embedding rank | A plausible regression that looked like "scaling doesn't work" | One SVD on 200 real embeddings |
| `mteb` cache serving stale scores | MTEB-14 identical to 2 dp across a changed model, while NanoBEIR moved 0.105 | Two metrics disagreeing about whether anything changed |
| String-replace edits silently not matching | Scripts ran, output looked plausible, plot showed old data | Switching to `Edit` (fails loudly) and asserting on match |

**The pattern:** in every case a *plausible number* was wrong for a mechanical
reason, and the tell was an internal inconsistency between two signals rather than
anything wrong-looking in the headline figure. Cross-checking two independent
signals caught all seven; trusting the headline would have caught none.
