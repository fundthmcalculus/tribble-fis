"""First-order TSK: rule consequents are word distributions, not scalars (E28).

Why the zero-order system has a floor
-------------------------------------
E27.5's conclusion. A zero-order TSK rule carries a **scalar** consequent, so
``IF prev1:DETERMINER AND cand:OPEN_NOUN THEN 0.265`` assigns the same 0.265 to every noun in
the vocabulary. The model can say "a noun follows" and never "which noun". Six refuted
hypotheses (corpus scale, OOV quality, rule budget, context width, significance gating,
relational slots) all failed to move that, because none of them touched it: the limit is the
shape of the consequent, not the antecedents, the features, or the data.

What a first-order consequent should be a function of
-----------------------------------------------------
Classic Takagi-Sugeno makes the consequent linear in the inputs, ``c_r(x) = a_r . x + b_r``.
Applied here that means linear in the *candidate's* feature vector -- and it would not help,
because those features are the same coarse categories the antecedent already uses. A rule
saying "prefer noun.person over noun.substance" is expressible in the zero-order system
already, just as two rules. That route buys parameter efficiency, not capability.

The capability that is missing is **word identity**, so the consequent has to be a function
over words:

    p(w | ctx)  =  sum_r  omega_r(ctx) . p_r(w)   /   sum_r omega_r(ctx)

where ``omega_r(ctx)`` is the rule's **context-side** firing strength and ``p_r(w)`` is the
word distribution observed when that rule fires. Each rule becomes a soft context class with
its own next-word distribution.

This is a **fuzzy class-based language model** -- the soft-membership analogue of class-based
n-grams (Brown, Della Pietra, deSouza, Lai & Mercer, *Class-Based n-gram Models of Natural
Language*, Computational Linguistics 18(4), 1992), where a word's class was a hard partition
found by clustering. Here classes are *named, overlapping, and graded*: a context belongs to
``prev1:DETERMINER`` to degree 0.9 and to ``prev1:=the`` to degree 1.0 at the same time, and
both contribute their word distributions in proportion.

Two consequences worth stating before any numbers
-------------------------------------------------
**The NCE inversion disappears.** ``p_r(w)`` are already normalised distributions over words,
so there is no ``p ∝ q(w).s/(1-s)`` step and no risk of double-counting the unigram prior --
the failure mode that made candidate-side lexicalisation hurt in E19.

**Part of any gain will be bigram-in-disguise, and must be controlled for.** A rule whose
context side is a lexeme identity, ``ctx:prev1:=the``, has ``p_r(w) = p(w | prev = the)``:
that *is* a bigram row. Rules with categorical context sides give genuinely backed-off
class-conditional rows. So the honest evaluation ablates identity-context rules and reports
what the categorical rules achieve alone -- otherwise "beats the bigram" could just mean
"contains the bigram".
"""

from __future__ import annotations

import numpy as np

from .corpus import Corpus

#: Worker state for ``ContextClassMiner._order2_parallel``. Populated in the parent before the
#: fork so children inherit the large arrays through copy-on-write; passing them as arguments
#: would pickle hundreds of megabytes per worker.
_PAR: dict = {}


def _order2_stripe(ks):
    """Runs in a forked child: score one stripe of order-2 seed blocks."""
    miner, Fa, wa, Fb_all, wb, seeds, P1, est_mass1 = _PAR["state"]
    try:                        # BLAS threads would only contend across processes (E23.3)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            return miner._order2(Fa, wa, Fb_all, wb, seeds, P1, est_mass1, ks)
    except ImportError:
        return miner._order2(Fa, wa, Fb_all, wb, seeds, P1, est_mass1, ks)


class FirstOrderConsequents:
    """Estimates and serves per-rule word distributions for a fitted ranker.

    Reuses the ranker's rule base unchanged -- the antecedents are exactly the ones the
    zero-order system mined, so this isolates the effect of the consequent's shape.
    """

    def __init__(self, ranker, vocab: list[str], counts: dict[str, int] | None = None,
                 alpha: float = 0.5, weighting: str = "firing",
                 exclude_identity_context: bool = False):
        """``alpha`` is unigram-interpolation mass per rule (a rule seen 10 times must not
        assert a peaked distribution). ``weighting`` selects ``omega_r``; see ``_omega``.

        ``exclude_identity_context=True`` drops rules whose context side is a lexeme
        identity dim, which is the control for "does this only work because it memorised a
        bigram?".
        """
        self.r = ranker
        self.f = ranker.f
        self.alpha = alpha
        self.weighting = weighting
        model = ranker.model_
        if model is None:
            raise RuntimeError("fit the ranker first")

        offset = ranker.cand_offset_
        names = ranker.feature_names_
        keep = []
        for j, rule in enumerate(model.rules_):
            ctx_idx = [fi for fi in rule.features if fi < offset]
            if not ctx_idx:
                continue        # candidate-only rule: its class is "everything", i.e. unigram
            if exclude_identity_context and any(
                    names[fi].split(":", 2)[2].startswith("=") for fi in ctx_idx):
                continue
            keep.append((j, np.asarray(ctx_idx, dtype=np.intp)))
        self.rule_ids = [j for j, _ in keep]
        self.ctx_cols = [c for _, c in keep]
        self.rules = [model.rules_[j] for j in self.rule_ids]
        self.n_ctx = np.array([len(c) for c in self.ctx_cols], dtype=np.float64)

        # Decodable vocabulary, matching the zero-order generator so perplexities compare.
        cand_vec = getattr(ranker, "cand_vector", self.f._token_vector)
        self.words = [w for w in vocab if cand_vec(w).sum() > 0]
        self.index = {w: i for i, w in enumerate(self.words)}
        w = np.array([max((counts or {}).get(t, 1), 1) for t in self.words], dtype=float)
        self.unigram = w / w.sum()
        self.P: np.ndarray | None = None

    # -- estimation --------------------------------------------------------

    def _ctx_firing(self, tokens: list[str], i: int) -> np.ndarray:
        """Context-side firing of every retained rule, for a prediction at position ``i``."""
        ctx = self.r._context_vector(tokens, i)
        return np.array([ctx[cols].prod() for cols in self.ctx_cols])

    def fit(self, corpus: Corpus, max_positions: int = 20000, seed: int = 11,
            verbose: bool = False):
        """Accumulate ``p_r(w)`` from firing-weighted next-word counts.

        Positive positions only: the question is "given this rule's class, what word actually
        came next", which the NCE negatives say nothing about. Firing-weighted rather than
        hard-assigned, so a context that belongs to a class to degree 0.4 contributes 0.4 of
        an observation -- that is the whole difference from hard Brown classes.
        """
        n_rules, n_vocab = len(self.rules), len(self.words)
        C = np.zeros((n_rules, n_vocab), dtype=np.float32)
        rng = np.random.default_rng(seed)
        sents = [s for s in corpus.sentences if len(s) > self.r.window]
        n = 0
        for si in rng.permutation(len(sents)):
            sent = sents[si]
            for i in range(self.r.window, len(sent)):
                wi = self.index.get(sent[i])
                if wi is None:
                    continue
                fire = self._ctx_firing(sent, i)
                live = np.nonzero(fire > 0)[0]
                if live.size:
                    C[live, wi] += fire[live]
                n += 1
                if n >= max_positions:
                    break
            if n >= max_positions:
                break

        totals = C.sum(axis=1, keepdims=True)
        # Interpolate each rule's distribution toward the unigram by a fixed pseudo-count
        # mass. A rule firing on 11 positions would otherwise assert p=1.0 on one word.
        self.P = (C + self.alpha * self.unigram[None, :].astype(np.float32)) / (
            totals + self.alpha)
        self.n_positions_ = n
        # Information gain of each class, in nats: how much knowing the class sharpens the
        # next-word distribution relative to the unigram. Used for weighting, and reported
        # because it is the number that says whether a named class is worth anything.
        h0 = float(-(self.unigram * np.log(np.maximum(self.unigram, 1e-12))).sum())
        P64 = self.P.astype(np.float64)
        h_r = -(P64 * np.log(np.maximum(P64, 1e-12))).sum(axis=1)
        self.info_gain_ = np.maximum(h0 - h_r, 0.0)
        self.entropy0_ = h0
        if verbose:
            print(f"  first-order: {n_rules} rule classes, {n} positions, "
                  f"{n_vocab} words, nnz={int((C > 0).sum())}")
        return self

    # -- prediction --------------------------------------------------------

    def _omega(self, fire: np.ndarray) -> np.ndarray:
        """Mixture weight per rule.

        ``firing`` is plain context firing. ``specific`` multiplies by the antecedent's
        arity, because a 1-term rule fires at up to 1.0 while a 2-term rule fires at the
        product of two memberships and is therefore systematically quieter despite being
        more informative -- the same bias ``order_quota`` exists to counter in rule
        selection. ``support`` additionally damps rules estimated from few observations.
        """
        if self.weighting == "firing":
            return fire
        if self.weighting == "specific":
            return fire * self.n_ctx
        if self.weighting == "support":
            sup = np.array([r.support for r in self.rules])
            return fire * self.n_ctx * np.sqrt(np.maximum(sup, 1.0))
        if self.weighting == "infogain":
            # Measured need (E28.1): the mixture was dominated by flat classes.
            # `IF prev2:DETERMINER` fires at 1.00 and its word distribution is essentially the
            # unigram (of 0.137, and 0.055, ...), while `IF prev1:=did -> not(0.783)` is sharp
            # and correct -- and both got weight 1.00, so hundreds of uninformative classes
            # averaged the useful ones away. Weighting by how much a class actually reduces
            # uncertainty about the next word is the direct fix, and it is the same quantity
            # that should have selected the classes in the first place.
            return fire * self.info_gain_
        raise ValueError(f"unknown weighting {self.weighting!r}")

    def distribution(self, tokens: list[str], hedge: float = 1.0) -> np.ndarray:
        """``p(w | context)`` as a firing-weighted mixture of rule word distributions."""
        if self.P is None:
            raise RuntimeError("call fit() first")
        fire = self._ctx_firing(tokens, len(tokens))
        omega = self._omega(fire)
        live = omega > 0
        if not live.any():
            return self.unigram.copy()
        p = omega[live] @ self.P[live]
        s = p.sum()
        p = p / s if s > 0 else self.unigram.copy()
        if hedge != 1.0:
            p = p ** hedge
            p = p / p.sum()
        return p

    def perplexity(self, corpus: Corpus, max_positions: int = 1000, seed: int = 7,
                   min_ctx: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        lo = self.r.window if min_ctx is None else min_ctx
        sents = [s for s in corpus.sentences if len(s) > lo]
        nll, n = 0.0, 0
        for si in rng.permutation(len(sents)):
            sent = sents[si]
            for i in range(lo, len(sent)):
                wi = self.index.get(sent[i])
                if wi is None:
                    continue
                p = self.distribution(sent[:i])
                nll -= float(np.log(max(p[wi], 1e-12)))
                n += 1
                if n >= max_positions:
                    break
            if n >= max_positions:
                break
        return {"perplexity": float(np.exp(nll / max(n, 1))), "n": n}

    def generate(self, prompt: list[str], n_tokens: int = 14, hedge: float = 1.0,
                 seed: int = 1, block_repeat: int = 2) -> list[str]:
        rng = np.random.default_rng(seed)
        out = list(prompt)
        for _ in range(n_tokens):
            p = self.distribution(out, hedge=hedge)
            for w in out[-block_repeat:] if block_repeat else ():
                k = self.index.get(w)
                if k is not None:
                    p[k] = 0.0
            s = p.sum()
            if s <= 0:
                break
            out.append(self.words[int(rng.choice(len(p), p=p / s))])
        return out

    # -- interpretability / size ------------------------------------------

    def sparse_parameters(self, top_m: int = 20) -> int:
        """Learned parameters if each rule keeps only its top ``top_m`` words.

        The dense table is ``n_rules x n_vocab``, which would abandon the smallness claim, so
        the size that should be quoted is the sparse one -- and it is also the only version a
        person could read.
        """
        return len(self.rules) * (top_m + 1)

    def explain(self, tokens: list[str], top_rules: int = 3, top_words: int = 6) -> str:
        """The firing classes behind a prediction, each with its word preferences."""
        fire = self._ctx_firing(tokens, len(tokens))
        omega = self._omega(fire)
        order = np.argsort(omega)[::-1][:top_rules]
        lines = [f"context: {' '.join(tokens[-6:])!r}"]
        for k in order:
            if omega[k] <= 0:
                continue
            row = self.P[k]
            best = np.argsort(row)[::-1][:top_words]
            ante = " AND ".join(n for n in self.rules[k].names if n.startswith("ctx:"))
            words = ", ".join(f"{self.words[i]}({row[i]:.3f})" for i in best)
            lines.append(f"  [w={omega[k]:.2f}] IF {ante}\n        -> {words}")
        return "\n".join(lines)


class ContextClassMiner:
    """Mines *context-only* fuzzy classes, scored by information gain about the next word.

    Why this is needed on top of ``FirstOrderConsequents`` (E28.1). Reusing the zero-order rule
    base gives classes that are all **single features**: ``must_include`` forces every rule to
    touch the candidate and ``max_order=2``, so the context side of every rule is exactly one
    term. There are no conjunctive context classes at all, which is why the ``firing`` and
    ``specific`` weightings scored identically -- ``n_ctx`` was constant at 1.

    Smoothing is not a detail here, it is the dominant parameter, and its optimum depends on
    what the model is *for* (E28.3). ``alpha`` is unigram pseudo-count mass per class:

    ====== ============== ==============
    alpha  standalone ppl  best mixture
    ====== ============== ==============
    0.5    601.1          **256.2**
    5      437.0          257.2
    50     352.9          264.5
    200    **343.5**      272.9
    ====== ============== ==============

    The two optima are at opposite ends, for a reason worth stating: heavy smoothing makes each
    class a reliable standalone predictor, while a mixture partner already supplies reliable
    mass and wants the fuzzy model's *sharp* class-conditional peaks instead. Default 50 is a
    compromise; set it low when mixing and high when serving alone.

    Undersmoothing was also the whole of the first version's standalone failure. At alpha=0.5
    the class ``IF prev2:=the AND prev1:QUANTIFIER`` predicted ``jackal`` at 0.105 after "the
    little", off a handful of firing observations -- and information gain *rewards* low entropy,
    so class selection is actively biased toward the under-observed classes that look sharp by
    accident. Same multiplicity trap as E26.1, in a new place.

    ``must_include`` was *correct* for scalar consequents: a context-only rule shifts every
    candidate's score by the same constant and cannot change a ranking. With word-level
    consequents that reasoning inverts -- a context-only rule is exactly a class with its own
    word distribution, and it is the only thing that can express "which noun". So the two
    design choices are coupled, and this miner drops the constraint.

    The objective changes with it. Lift is meaningless for a context-only antecedent under the
    NCE target (zero marginal lift by construction, as ``seed_features`` documents). The right
    objective for a class-based model is **firing-weighted information gain** about the next
    word, ``mass_r * (H(unigram) - H(p_r))``, which is the standard decision-tree criterion
    applied to fuzzy memberships. Multiplying by mass is what stops a class that fires on nine
    positions with one word after it from outranking a real generalisation.
    """

    def __init__(self, ranker, vocab: list[str], counts: dict[str, int] | None = None,
                 alpha: float = 50.0, top_singles: int = 140, max_classes: int = 3000,
                 max_order: int = 2, min_mass: float = 20.0,
                 exclude_identity: bool = False, selection_holdout: float = 0.0,
                 backoff: str = "unigram", pair_chunk: int = 600,
                 dtype=np.float32, n_jobs: int = 1):
        self.r = ranker
        self.f = ranker.f
        self.alpha = alpha
        self.top_singles = top_singles
        self.max_classes = max_classes
        self.max_order = max_order
        self.min_mass = min_mass
        self.exclude_identity = exclude_identity
        # E29.1: score candidate classes on rows that did NOT estimate them. In-sample
        # information gain rewards low entropy, so it systematically prefers classes that
        # happen to look sharp on few observations -- the `jackal` failure (E28.3). Held-out
        # cross-entropy reduction cannot be gamed that way, because a class sharpened by
        # accident predicts held-out words no better than the unigram does.
        self.selection_holdout = selection_holdout
        # E29.2: what a class's distribution is smoothed *toward*. "unigram" throws away the
        # hierarchy the miner just built -- `prev2:=the AND prev1:QUANTIFIER` should fall back
        # on `prev1:QUANTIFIER`, not on corpus frequency. "parent" is the standard remedy
        # (Katz backoff, Kneser-Ney).
        self.backoff = backoff
        # How many order-2 candidate columns to materialise at once. Purely a memory bound.
        self.pair_chunk = pair_chunk
        # Working precision of the position x feature matrix and the block counts (E31 item 3).
        # float32 halves both memory and bandwidth, and after item 2 the profile is dominated by
        # copies rather than arithmetic. `F` at full data is 2.6 GB in float64.
        #
        # This is NOT bit-exact: counts are sums of products, so float32 carries ~1e-4 relative
        # error. It is the default only because held-out perplexity was measured to be unchanged
        # to four decimal places (422.6558 either way); the distributions themselves are formed
        # and scored in float64, where the arrays are small. Pass float64 to rule it out.
        self.dtype = dtype
        # Processes for the order-2 block sweep (E31 item 4b). Parallelised over *seed blocks*
        # rather than rows: row-sharding would make each worker return partial counts for every
        # candidate pair before anything could be filtered (~112 MB per worker), whereas a seed
        # stripe returns only the classes that survive `min_mass`. `fork` gives workers `F`
        # through copy-on-write, so nothing large is pickled inward. Exact -- the same blocks,
        # computed elsewhere. Default 1 (serial): after items 2-3 this stage is ~12s of a ~46s
        # pipeline, so the win is real but modest, and forking in a library default is not.
        self.n_jobs = n_jobs
        self.weighting = "infogain"
        cand_vec = getattr(ranker, "cand_vector", self.f._token_vector)
        self.words = [w for w in vocab if cand_vec(w).sum() > 0]
        self.index = {w: i for i, w in enumerate(self.words)}
        w = np.array([max((counts or {}).get(t, 1), 1) for t in self.words], dtype=float)
        self.unigram = w / w.sum()
        self.P: np.ndarray | None = None

    # -- estimation helpers ------------------------------------------------

    def _counts(self, F: np.ndarray, w_idx: np.ndarray) -> np.ndarray:
        """``C[j, w] = sum_i F[i, j] . 1[w_i = w]``.

        A sparse one-hot GEMM, the same reformulation as ``_pair_blocks`` -- which is where it
        was applied first, leaving *this* (the order-1 path) as the last ``np.add.at`` in the
        pipeline and 19% of full-data class estimation (E31 item 4d). ``np.add.at`` is unbuffered
        and slow by construction; ``Y.T @ F`` dispatches to a sparse kernel.
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            CT = np.zeros((len(self.words), F.shape[1]), dtype=self.dtype)
            np.add.at(CT, w_idx, F)
            return CT.T
        Yt = sp.csr_matrix(
            (np.ones(len(w_idx), dtype=F.dtype), (w_idx, np.arange(len(w_idx)))),
            shape=(len(self.words), len(w_idx)))
        return np.ascontiguousarray((Yt @ F).T)

    def _smooth(self, C: np.ndarray, prior: np.ndarray) -> tuple:
        """Interpolate counts toward ``prior``. ``prior`` is per-class when backing off."""
        C = np.asarray(C, dtype=np.float64)     # blocks are small; logs want the precision
        mass = C.sum(axis=1)
        if prior.ndim == 1:
            prior = prior[None, :]
        P = (C + self.alpha * prior) / (mass[:, None] + self.alpha)
        return P, mass

    def _held_out_gain(self, P: np.ndarray, Fb: np.ndarray, wb: np.ndarray) -> np.ndarray:
        """Cross-entropy reduction over the unigram, on rows that did not estimate ``P``.

        For class ``j``, the firing-weighted mean of ``log p_j(w) - log unigram(w)`` over
        held-out rows, times held-out mass. A class sharpened by accident scores near zero
        here because its peak lands on words the held-out rows do not contain -- which is
        exactly what in-sample entropy could not detect (E28.3).
        """
        ratio = np.log(np.maximum(P, 1e-12)) - np.log(np.maximum(self.unigram, 1e-12))[None, :]
        # per-row log-ratio for each class, firing-weighted: (n_rows, n_classes)
        per_row = ratio[:, wb].T                      # (n_rows, n_classes)
        num = (Fb * per_row).sum(axis=0)
        mass_b = Fb.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_gain = np.where(mass_b > 0, num / np.maximum(mass_b, 1e-12), 0.0)
        return mass_b, np.maximum(mean_gain, 0.0)

    def _score(self, C: np.ndarray, prior: np.ndarray, Fb, wb) -> tuple:
        """Estimate distributions, then score them -- in sample or held out."""
        P, mass = self._smooth(C, prior)
        h0 = float(-(self.unigram * np.log(np.maximum(self.unigram, 1e-12))).sum())
        if Fb is None:
            h = -(P * np.log(np.maximum(P, 1e-12))).sum(axis=1)
            gain = np.maximum(h0 - h, 0.0)
            return P, mass, gain, h0
        mass_b, gain = self._held_out_gain(P, Fb, wb)
        # Selection mass is the held-out mass, so a class that never fires on held-out rows
        # cannot be selected on the strength of its training fit.
        return P, mass_b, gain, h0

    def _pair_blocks(self, Fa, wa, Fb_all, seeds, wb_all=None, ks=None):
        """Yield ``(pairs, counts, holdout_columns, holdout_words)`` per order-2 block.

        The counting problem is a three-way contraction::

            C[(a,b), w] = sum_i  F[i,a] . F[i,b] . 1[w_i = w]

        The obvious implementation forms one column per pair and scatter-adds it, and
        profiling says that is where essentially all of training time goes: at 60,000
        positions, 40% in the per-pair column products, 34% in ``column_stack``, 20% in
        ``np.add.at`` -- 94% total, against 1% for collecting ``F`` (E30).

        Fixing the *shape* rather than the speed: hold one seed ``a`` fixed and the whole row
        of pairs ``(a, b)`` for every later ``b`` becomes a single scaled matrix times a
        sparse one-hot matrix::

            G = F[:, later_seeds] * F[:, a]          (one block multiply)
            C = (Y.T @ G).T                          (one sparse GEMM)

        Identical arithmetic, ~140 block operations instead of ~9,730 slice multiplies plus
        17 giant ``column_stack`` allocations, and the scatter-add disappears because the
        counts fall out of the GEMM. This is the same reformulation that made rule growth
        affordable in E14, applied to the other half of the pipeline.
        """
        try:
            import scipy.sparse as sp
        except ImportError:                     # keep working without scipy
            for pairs, C, F2b in self._pair_blocks_reference(Fa, wa, Fb_all, seeds):
                yield pairs, C, F2b, wb_all
            return

        n_vocab = len(self.words)
        S = np.asarray(seeds, dtype=np.intp)
        Fs = Fa[:, S]
        Fsb = None if Fb_all is None else Fb_all[:, S]
        for k in (range(len(S) - 1) if ks is None else ks):
            pairs = [(int(S[k]), int(b)) for b in S[k + 1:]]
            # E31 item 2: only rows where the FIXED seed fires can contribute to any pair
            # (a, b) -- a product is zero if either factor is. `F` is ~1.5% dense, so the
            # unrestricted block was a dense GEMM over almost entirely zero rows. Exact.
            idx = np.nonzero(Fs[:, k])[0]
            if idx.size == 0:
                continue                        # every pair with this seed has zero mass
            Fk = Fs[idx]
            G = Fk[:, k + 1:] * Fk[:, k:k + 1]
            # Per-block one-hot over just these rows. Built fresh rather than slicing a
            # shared CSR, because column-slicing is the slow axis for CSR and this is O(nnz).
            Yk = sp.csr_matrix(
                (np.ones(idx.size, dtype=G.dtype), (wa[idx], np.arange(idx.size))),
                shape=(n_vocab, idx.size))
            C = np.ascontiguousarray((Yk @ G).T)
            F2b, wb_blk = None, wb_all
            if Fsb is not None:
                idxb = np.nonzero(Fsb[:, k])[0]
                Fkb = Fsb[idxb]
                F2b = Fkb[:, k + 1:] * Fkb[:, k:k + 1]
                wb_blk = wb_all[idxb]
            yield pairs, C, F2b, wb_blk

    def _order2(self, Fa, wa, Fb_all, wb, seeds, P1, est_mass1, ks):
        """Score every order-2 block in ``ks`` and return the classes that survive.

        Extracted from ``fit`` so the sweep can be mapped over seed stripes by
        ``_order2_parallel`` without duplicating the scoring logic.
        """
        out = []
        for pairs, Ca, F2b_pre, wb_blk in self._pair_blocks(
                Fa, wa, Fb_all, seeds, wb_all=wb, ks=ks):
            # E29.2: back off to the parents, not the unigram. A conjunction's parents are the
            # two singles it grew from, and they are far more informative about the next word
            # than corpus frequency. Mass-weighted so the better-estimated parent dominates.
            if self.backoff == "parent":
                pa = np.array([est_mass1[a] for a, _ in pairs])
                pb = np.array([est_mass1[b] for _, b in pairs])
                wsum = np.maximum(pa + pb, 1e-12)
                # P1 rows are indexed by FEATURE ID (it was built over all context columns),
                # not by position in `ok`. Indexing it by rank silently mixes up whose parent
                # is whose.
                prior2 = ((pa / wsum)[:, None] * P1[[a for a, _ in pairs]]
                          + (pb / wsum)[:, None] * P1[[b for _, b in pairs]])
            elif self.backoff == "unigram":
                prior2 = self.unigram
            else:
                raise ValueError(f"unknown backoff {self.backoff!r}")
            P2, sel_mass2, gain2, _ = self._score(Ca, prior2, F2b_pre, wb_blk)
            est_mass2 = Ca.sum(axis=1)
            keep = np.nonzero((sel_mass2 >= self.min_mass)
                              & (est_mass2 >= self.min_mass))[0]
            # Cast here, in both paths, so serial and parallel results are identical -- `P` is
            # stored as float32 regardless, and this keeps the inter-process payload small.
            out += [(pairs[i], P2[i].astype(np.float32), sel_mass2[i], gain2[i])
                    for i in keep]
        return out

    def _order2_parallel(self, Fa, wa, Fb_all, wb, seeds, P1, est_mass1):
        """Run the order-2 sweep across processes, striping the seed index.

        Striped (``i::n``) rather than contiguous, because block cost shrinks with ``k`` -- the
        first seed pairs with 139 others and the last with one, so contiguous chunks would leave
        one worker with almost all the work.
        """
        import multiprocessing as mp
        n_jobs = mp.cpu_count() if self.n_jobs < 0 else self.n_jobs
        n_blocks = len(seeds) - 1
        if n_jobs <= 1 or n_blocks <= 1:
            return self._order2(Fa, wa, Fb_all, wb, seeds, P1, est_mass1, None)
        _PAR["state"] = (self, Fa, wa, Fb_all, wb, seeds, P1, est_mass1)
        stripes = [list(range(i, n_blocks, n_jobs)) for i in range(n_jobs)]
        stripes = [st for st in stripes if st]
        try:
            with mp.get_context("fork").Pool(n_jobs) as pool:
                parts = pool.map(_order2_stripe, stripes)
        finally:
            _PAR.clear()
        return [cls for part in parts for cls in part]

    def _pair_blocks_reference(self, Fa, wa, Fb_all, seeds):
        """The original per-pair implementation. Kept as a correctness oracle."""
        all_pairs = [(int(a), int(b)) for i, a in enumerate(seeds)
                     for b in seeds[i + 1:]]
        for lo in range(0, len(all_pairs), self.pair_chunk):
            pairs = all_pairs[lo: lo + self.pair_chunk]
            C = self._counts(np.column_stack([Fa[:, a] * Fa[:, b] for a, b in pairs]), wa)
            F2b = (None if Fb_all is None else
                   np.column_stack([Fb_all[:, a] * Fb_all[:, b] for a, b in pairs]))
            yield pairs, C, F2b

    def fit(self, corpus: Corpus, max_positions: int = 20000, seed: int = 11,
            verbose: bool = False):
        names = self.r.feature_names_ or self.r._names()
        offset = self.r.cand_offset_
        rng = np.random.default_rng(seed)
        sents = [s for s in corpus.sentences if len(s) > self.r.window]

        # Two passes so `F` can be allocated exactly and filled in place. Collecting rows
        # into a Python list first held 624K separate arrays *and* their concatenation at the
        # same time -- 2.6 GB transiently, and measured peak RSS 3.95 GB (E31 item 4a). The
        # counting pass is only dict lookups, so it is nearly free.
        order = rng.permutation(len(sents))
        n = 0
        for si in order:
            sent = sents[si]
            for i in range(self.r.window, len(sent)):
                if self.index.get(sent[i]) is not None:
                    n += 1
                    if n >= max_positions:
                        break
            if n >= max_positions:
                break
        F = np.empty((n, self.r.cand_offset_), dtype=self.dtype)
        w_idx = np.empty(n, dtype=np.intp)
        r = 0
        for si in order:
            sent = sents[si]
            for i in range(self.r.window, len(sent)):
                wi = self.index.get(sent[i])
                if wi is None:
                    continue
                F[r] = self.r._context_vector(sent, i)[:self.r.cand_offset_]
                w_idx[r] = wi
                r += 1
                if r >= n:
                    break
            if r >= n:
                break
        n = r
        # NOT table-gathered, deliberately (E31 item 4c, reverted). E23's table trick was tried
        # here after item 4b's Amdahl analysis suggested F construction was the serial majority.
        # Measured in isolation at 300,000 positions: per-position 1.31s, table gather with a
        # strided `out=` 1.28s (the strided write cancels the gain), table gather with `hstack`
        # 0.92s -- but `hstack` needs every slot materialised at once, 2.6 GB transient at full
        # data, which would undo item 4a's memory fix to save 0.4s. And the premise was wrong
        # anyway: F construction is 1.3s of a 5.9s fit, not the serial majority.
        if self.exclude_identity:
            drop = [k for k in range(offset)
                    if names[k].split(":", 2)[2].startswith("=")]
            F[:, drop] = 0.0        # F is freshly allocated and context-only; no copy needed

        # Split rows: A estimates the distributions, B scores them (E29.1). Without this,
        # gain is measured on the rows that produced it.
        if self.selection_holdout > 0:
            rs = np.random.default_rng(seed + 1)
            in_b = rs.random(F.shape[0]) < self.selection_holdout
            Fa, wa = F[~in_b], w_idx[~in_b]
            Fb_all, wb = F[in_b], w_idx[in_b]
        else:
            Fa, wa, Fb_all, wb = F, w_idx, None, None

        # Order 1: every context feature is a candidate class.
        C1 = self._counts(Fa, wa)
        Fb1 = Fb_all
        P1, sel_mass1, gain1, h0 = self._score(C1, self.unigram, Fb1, wb)
        est_mass1 = C1.sum(axis=1)
        score1 = sel_mass1 * gain1
        ok = np.nonzero((sel_mass1 >= self.min_mass) & (est_mass1 >= self.min_mass))[0]
        ok = ok[np.argsort(score1[ok])[::-1]]
        classes = [((int(j),), P1[j], sel_mass1[j], gain1[j]) for j in ok]

        # Order 2: conjunctions among the strongest singles. The pool is bounded by
        # `top_singles` because the pair count is quadratic, and E26.1's lesson is that a
        # larger search space is not free.
        if self.max_order >= 2 and len(ok):
            seeds = ok[: self.top_singles]
            all_pairs = [(int(a), int(b)) for i, a in enumerate(seeds)
                         for b in seeds[i + 1:]]
            # Chunked, because materialising every pair column at once is what actually
            # limits this: 150,000 positions x 9,730 pairs x 8 bytes is 11.7 GB, and the
            # first attempt was SIGKILLed by the OOM killer with no traceback (E29.3).
            # Chunking bounds peak memory at chunk_size x n_positions instead.
            if self.n_jobs == 1:
                classes += self._order2(Fa, wa, Fb_all, wb, seeds, P1, est_mass1, None)
            else:
                classes += self._order2_parallel(Fa, wa, Fb_all, wb, seeds, P1, est_mass1)

        classes.sort(key=lambda t: -(t[2] * t[3]))
        classes = classes[: self.max_classes]
        self.ctx_cols = [np.asarray(c[0], dtype=np.intp) for c in classes]
        self.P = np.asarray([c[1] for c in classes], dtype=np.float32)
        self.mass_ = np.asarray([c[2] for c in classes])
        self.info_gain_ = np.asarray([c[3] for c in classes])
        self.n_ctx = np.asarray([len(c[0]) for c in classes], dtype=np.float64)
        self.entropy0_ = h0
        self.class_names = [" AND ".join(names[k] for k in c[0]) for c in classes]
        self.n_positions_ = n
        if verbose:
            print(f"  mined {len(classes)} context classes from {offset} features, "
                  f"{n} positions; best gain {self.info_gain_[:3]}")
        return self

    # Prediction/serving is identical to FirstOrderConsequents, so borrow it rather than
    # duplicating -- a second copy of the mixture arithmetic is exactly how the generator and
    # the ranker drifted apart in E27.
    _ctx_firing = FirstOrderConsequents._ctx_firing
    _omega = FirstOrderConsequents._omega
    distribution = FirstOrderConsequents.distribution
    perplexity = FirstOrderConsequents.perplexity
    generate = FirstOrderConsequents.generate

    @property
    def rules(self):
        """Alias so borrowed methods that inspect ``rules`` keep working."""
        return self.ctx_cols

    def sparse_parameters(self, top_m: int = 20) -> int:
        return len(self.ctx_cols) * (top_m + 1)

    def explain(self, tokens: list[str], top_rules: int = 3, top_words: int = 6) -> str:
        fire = self._ctx_firing(tokens, len(tokens))
        omega = self._omega(fire)
        order = np.argsort(omega)[::-1][:top_rules]
        lines = [f"context: {' '.join(tokens[-6:])!r}"]
        for k in order:
            if omega[k] <= 0:
                continue
            row = self.P[k]
            best = np.argsort(row)[::-1][:top_words]
            words = ", ".join(f"{self.words[i]}({row[i]:.3f})" for i in best)
            lines.append(f"  [w={omega[k]:.2f} gain={self.info_gain_[k]:.2f} nats] "
                         f"IF {self.class_names[k]}\n        -> {words}")
        return "\n".join(lines)
