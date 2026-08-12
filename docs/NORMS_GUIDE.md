# Choosing the fuzzy operators

Every model here combines memberships with two connectives: a **t-norm** for the
rule AND, and a **t-conorm** for the per-feature OR. This guide covers how to
select them and which models honour the selection.

## The families

Five are available. Each name selects a *family*, whose t-norm and t-conorm are
De Morgan duals under the standard negation `N(x) = 1 - x`, i.e.
`S(x, y) = 1 - T(1-x, 1-y)`.

| Family | t-norm `T(x, y)` | t-conorm `S(x, y)` | Character |
|---|---|---|---|
| `min/max` *(default)* | `min(x, y)` | `max(x, y)` | Gödel. Idempotent — only the weakest feature matters, so corroborating evidence is discarded |
| `probability` | `xy` | `x + y - xy` | Strict and smooth; many moderate memberships accumulate |
| `luk` | `max(0, x+y-1)` | `min(1, x+y)` | Łukasiewicz. Nilpotent — reaches exactly 0, sparsifying the rule base |
| `hamacher` | `xy / (x+y-xy)` | `(x+y-2xy) / (1-xy)` | Strictest of the five; sharp at low membership. Has removable singularities, guarded internally |
| `einstein` | `xy / (2-(x+y-xy))` | `(x+y) / (1+xy)` | Sits between Łukasiewicz and the algebraic product. Both denominators lie in `[1, 2]`, so it is singularity-free |

Ordering at a glance: `luk ≤ einstein ≤ probability ≤ min/max` for the t-norm,
reversed for the conorm.

## Normal use

Pass a family and both halves come from it:

```python
TribbleRegressor(norm_conorm="einstein")
TribbleClassifier(norm_conorm="hamacher")
AnomalyParameters(include_anomaly=True, norm_conorm="probability")
```

## Advanced: mixing families

`t_norm` and `t_conorm` override one half each. A pair drawn from two different
families is **not** a De Morgan dual pair, so it is rejected unless you also pass
`allow_mixed_norms=True`:

```python
TribbleRegressor(t_norm="probability", t_conorm="luk")
# ValueError: ... are from different families and so are not De Morgan duals.

TribbleRegressor(t_norm="probability", t_conorm="luk",
                                 allow_mixed_norms=True)      # fine
```

The gate exists because the anomaly rule is a De Morgan construction — it scores
the unknown class as `1 - S(mu_1, ..., mu_k)`, the complement of the aggregate of
the known-class rules. That reads as "nothing known fires" only while `S` is the
dual of the `T` used to build those firings. With a mismatched pair the rule still
computes, but it no longer means what its derivation says it means. Mixing is
therefore supported for experiments and refused by accident.

Resolution is available directly if you need it:

```python
from tribblefis.gauss_data import resolve_norm_pair
resolve_norm_pair("einstein")                     # NormPair('einstein', 'einstein')
resolve_norm_pair().is_de_morgan                  # True
```

## Which models honour the selection

| Model | Parameter | Default |
|---|---|---|
| `TribbleRegressor` | `norm_conorm`, `t_norm`, `t_conorm` | `min/max` |
| `TribbleClassifier` | `norm_conorm`, `t_norm`, `t_conorm` | `min/max` |
| `TribbleSequenceClassifier` | `norm_conorm`, `t_norm`, `t_conorm` | `min/max` |
| `AnomalyParameters` | `norm_conorm`, `t_norm`, `t_conorm` | `min/max` |
| Ruspini models | `norm_conorm` | `probability` |
| `FuzzyRegressionTree`, `FuzzyClassificationTree` | `t_norm` (AND only — paths have no OR) | `probability` |

Low-level callers can pass a resolved pair straight through:

```python
tsk_firing_strengths(X, model, norms=resolve_norm_pair("luk"))
solve_tsk_consequents(..., norms=resolve_norm_pair("luk"))
predict_tsk(..., norms=resolve_norm_pair("luk"))
```

### The one model that does not take a t-norm

**`HierarchicalFuzzyExpertsRegressor` / `...Classifier` (HME) gates are always a
product, and that is deliberate.** Each gate node normalises its memberships to
sum to 1, and multiplying those factors along a root-to-leaf path keeps the leaf
responsibilities summing to 1. That partition of unity is what makes the model a
mixture of experts — the leaf weights *are* mixture weights, and the blend is
convex. `min`, Łukasiewicz or Hamacher would each break the normalisation and the
leaves would stop being a distribution. So the HME gate is fixed by the model's
semantics rather than being a free axis; only its experts, which are ordinary MoG
sub-models, take a norm.
