"""FES — a text embedding model whose forward pass is a first-order TSK fuzzy
inference system.

See ``../DESIGN.md`` for the derivation. The short version:

    token ids -> narrow feature table F           (fuzzification)
              -> HTSK rule firing over R rules    (antecedent, softmax over -dist^2)
              -> mixture of R linear experts      (first-order TSK consequent)
              -> learned SIF-style weighted pool  (defuzzification over the sequence)
              -> L2 normalise

The one non-obvious requirement is HTSK. TSK defuzzification with Gaussian MFs and
a product t-norm *is* ``softmax(Z)`` with ``Z_r = -sum_d (v_d - m_rd)^2/(2 s_rd^2)``,
whose magnitude grows as O(D). At D=64 that saturates the softmax, one rule wins
every token, and the antecedent stops learning. Dividing the exponent by D — the
geometric-mean t-norm — removes the D-dependence.

    Cui, Wu & Xu, "Curse of Dimensionality for TSK Fuzzy Neural Networks:
    Explanation and Solutions", IJCNN 2021. arXiv:2102.04271
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

from sentence_transformers.base.modules.input_module import InputModule

# ``defuzzify`` choices. "htsk" is the default and the only one we recommend.
DEFUZZ_HTSK = "htsk"  # softmax(Z / D)      -- geometric-mean t-norm
DEFUZZ_PROD = "product"  # softmax(Z)          -- vanilla; saturates, kept for ablation A3
DEFUZZ_LOG = "logtsk"  # l1-normalise -1/Z   -- LogTSK


class FuzzyEmbedding(InputModule):
    """Sentence-Transformers input module implementing the FES forward pass.

    Args:
        tokenizer: a fast (Rust-backed) tokenizer.
        d_in: width of the token feature table ``F``; the antecedent input space.
        d_out: output embedding dimension.
        n_rules: number of fuzzy rules ``R``. ``n_rules=1`` reduces the model to a
            low-rank static embedding model and is the control ablation.
        consequent_order: ``1`` for TSK order-1 (``A_r v + b_r``), ``0`` for
            order-0 (``b_r`` only, i.e. a Sugeno constant consequent).
        context_conditioned: if True (variant FES-C), rules fire on
            ``[v_t ; c]`` where ``c`` is the pooled document feature vector, so a
            token routes to different experts in different documents. This is the
            only way to exceed the bag-of-words ceiling.
        learn_temperature: make the HTSK divisor a learned scalar (init ``log D``).
        feature_norm: LayerNorm the token features before firing. HTSK's
            scale-freeness assumes roughly standardised inputs.
        rule_dropout: DropRule probability, applied to firing strengths in training.
        max_seq_length: truncation length.
    """

    config_keys: list[str] = [
        "d_in",
        "d_out",
        "n_rules",
        "consequent_order",
        "context_conditioned",
        "defuzzify",
        "learn_temperature",
        "feature_norm",
        "rule_dropout",
        "max_seq_length",
        "pool",
        "consequent_rank",
    ]
    config_file_name: str = "fuzzy_embedding_config.json"
    modalities: list[str] = ["text"]

    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        d_in: int = 64,
        d_out: int = 256,
        n_rules: int = 32,
        consequent_order: int = 1,
        context_conditioned: bool = False,
        defuzzify: str = DEFUZZ_HTSK,
        learn_temperature: bool = True,
        feature_norm: bool = True,
        rule_dropout: float = 0.0,
        max_seq_length: int = 256,
        pool: str = "learned",  # "learned" (SIF-style) | "mean"
        consequent_rank: int | None = None,
        sigma_init: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            tokenizer = tokenizer._tokenizer
        elif not isinstance(tokenizer, Tokenizer):
            raise ValueError(
                "The tokenizer must be fast (Rust-backed). Use Tokenizer.from_pretrained()."
            )
        if defuzzify not in (DEFUZZ_HTSK, DEFUZZ_PROD, DEFUZZ_LOG):
            raise ValueError(f"unknown defuzzify={defuzzify!r}")
        if consequent_order not in (0, 1):
            raise ValueError("consequent_order must be 0 or 1")

        self.tokenizer = tokenizer
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

        self.vocab_size = tokenizer.get_vocab_size()
        self.d_in = d_in
        self.d_out = d_out
        self.n_rules = n_rules
        self.consequent_order = consequent_order
        self.context_conditioned = context_conditioned
        self.defuzzify = defuzzify
        self.learn_temperature = learn_temperature
        self.feature_norm = feature_norm
        self.rule_dropout = rule_dropout
        self.max_seq_length = max_seq_length
        self.pool = pool

        # --- fuzzification: the narrow per-token feature table -----------------
        # This is the only per-vocabulary-item parameter block of any width, and
        # the reason the model is small: d_in << d_out.
        self.features = nn.Embedding(self.vocab_size, d_in)
        nn.init.normal_(self.features.weight, std=d_in**-0.5)

        # Learned per-token pooling logit. Initialised to 0 and (optionally)
        # overwritten with SIF weights from corpus frequencies; see
        # ``init_pool_weights_from_frequency``.
        self.pool_logit = nn.Embedding(self.vocab_size, 1)
        nn.init.zeros_(self.pool_logit.weight)

        self.norm = nn.LayerNorm(d_in) if feature_norm else nn.Identity()

        # --- antecedent: R rules, each a diagonal Gaussian over D dims ---------
        # D = 2*d_in for FES-C (token features concatenated with context).
        self.D = 2 * d_in if context_conditioned else d_in
        self.centers = nn.Parameter(torch.randn(n_rules, self.D) * 0.5)
        # sigma stored as log sigma, guaranteeing positivity.
        self.log_sigma = nn.Parameter(torch.full((n_rules, self.D), math.log(sigma_init)))

        # HTSK divisor as a learned scalar in log space; exp(log_tau) == D at init,
        # so the model *starts* at exact HTSK and may anneal from there.
        init_log_tau = math.log(self.D) if defuzzify == DEFUZZ_HTSK else 0.0
        self.log_tau = nn.Parameter(
            torch.tensor(init_log_tau), requires_grad=bool(learn_temperature)
        )

        # --- consequent: R local linear experts, d_in -> d_out -----------------
        # Dense costs R * d_in * d_out, which is what caps R: at d_in=64, d_out=256
        # going R=32 -> 512 takes the model from 2.5M to 10.6M parameters.
        #
        # consequent_rank = k factorises A_r = U_r @ V with V shared across rules,
        # costing R*d_in*k + k*d_out instead. R=512 then costs 3.2M rather than
        # 10.6M, and d_out becomes nearly free (768-d at R=64: 2.20M vs 5.19M).
        # Each rule keeps its own d_in -> k map, so rules stay individually
        # interpretable; what they share is the k -> d_out output basis.
        self.consequent_rank = consequent_rank
        if consequent_rank and consequent_rank + n_rules < d_out:
            # V is shared across rules, so the model's entire output lives in a
            # (k + min(R, d_out))-dimensional subspace of R^d_out. Setting k well
            # below d_out yields padded low-rank vectors: E012 measured a rank-32
            # model emitting 512-d embeddings with an effective rank of 65, losing
            # 3.5 MTEB-14 and 0.095 NanoBEIR against a dense d_out=256 model.
            warnings.warn(
                f"consequent_rank={consequent_rank} with n_rules={n_rules} caps the "
                f"embedding rank at ~{consequent_rank + n_rules} but d_out={d_out}. "
                "The output will be low-rank padding. Raise consequent_rank toward "
                "d_out, or use a dense consequent.",
                UserWarning, stacklevel=2,
            )
        if consequent_order == 1:
            if consequent_rank:
                self.expert_u = nn.Parameter(torch.empty(n_rules, d_in, consequent_rank))
                nn.init.normal_(self.expert_u, std=d_in**-0.5)
                self.shared_v = nn.Parameter(torch.empty(consequent_rank, d_out))
                nn.init.normal_(self.shared_v, std=consequent_rank**-0.5)
                self.register_parameter("expert_w", None)
            else:
                self.expert_w = nn.Parameter(torch.empty(n_rules, d_in, d_out))
                nn.init.normal_(self.expert_w, std=d_in**-0.5)
                self.register_parameter("expert_u", None)
                self.register_parameter("shared_v", None)
        else:
            self.register_parameter("expert_w", None)
            self.register_parameter("expert_u", None)
            self.register_parameter("shared_v", None)
        self.expert_b = nn.Parameter(torch.zeros(n_rules, d_out))
        nn.init.normal_(self.expert_b, std=d_out**-0.5)

        self._drop = nn.Dropout(rule_dropout) if rule_dropout > 0 else nn.Identity()

        # Diagnostics filled in by the forward pass, consumed by the UR loss and
        # by the rule-usage entropy metric. Not parameters, not saved.
        self.last_firing_mean: torch.Tensor | None = None
        self.last_token_entropy: torch.Tensor | None = None

        self.base_model = kwargs.get("base_model", None)

    # ------------------------------------------------------------------ params
    def parameter_counts(self) -> dict[str, int]:
        """Exact parameter accounting, for the results tables."""
        counts = {
            "token_features": self.features.weight.numel(),
            "pool_logit": self.pool_logit.weight.numel(),
            "antecedents": self.centers.numel() + self.log_sigma.numel(),
            "temperature": 1,
            "consequent_bias": self.expert_b.numel(),
        }
        counts["consequent_weight"] = (
            self.expert_w.numel() if self.expert_w is not None else 0
        )
        if self.expert_u is not None:
            counts["consequent_lowrank_u"] = self.expert_u.numel()
            counts["consequent_shared_v"] = self.shared_v.numel()
        if self.feature_norm:
            counts["layer_norm"] = 2 * self.d_in
        counts["total"] = sum(counts.values())
        return counts

    def param_groups(
        self, sparse_lr: float, dense_lr: float, temp_lr: float | None = None
    ) -> list[dict]:
        """Three learning-rate groups.

        - **sparse tables** want ~0.2 (the static-embedding recipe's LR).
        - **dense antecedent/consequent** parameters want ~1e-3.
        - **the HTSK temperature** is a single scalar controlling the sharpness of
          every routing decision in the model. At the dense LR it moves by ~0.07
          over a whole epoch, which is far too slow to escape a bad
          initialisation. It gets its own much larger LR.

        Using one LR for all three either diverges or crawls, so this split is not
        optional.
        """
        sparse = [self.features.weight, self.pool_logit.weight]
        temp = [self.log_tau] if self.log_tau.requires_grad else []
        excluded = {id(p) for p in sparse} | {id(p) for p in temp}
        dense = [p for p in self.parameters() if id(p) not in excluded and p.requires_grad]
        groups = [
            {"params": sparse, "lr": sparse_lr, "name": "sparse_tables"},
            {"params": dense, "lr": dense_lr, "name": "dense_fis"},
        ]
        if temp:
            groups.append(
                {"params": temp, "lr": temp_lr if temp_lr is not None else 50 * dense_lr,
                 "name": "htsk_temperature"}
            )
        return groups

    # ------------------------------------------------------------- tokenisation
    def preprocess(
        self, inputs: list[str], prompt: str | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        encodings = self.tokenizer.encode_batch(inputs, add_special_tokens=False)
        ids = [e.ids[: self.max_seq_length] or [0] for e in encodings]
        width = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), width), dtype=np.int64)
        mask = np.zeros((len(ids), width), dtype=np.float32)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = row
            mask[i, : len(row)] = 1.0
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(mask),
        }

    # ------------------------------------------------------------------ forward
    def firing_strengths(
        self, v: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalised rule firing strengths ``f_bar``.

        Args:
            v: ``(B, L, d_in)`` token features (already normalised).
            context: ``(B, 1, d_in)`` document context, for FES-C.

        Returns:
            ``(B, L, R)`` firing strengths, summing to 1 over the rule axis.
        """
        if context is not None:
            v = torch.cat([v, context.expand(-1, v.shape[1], -1)], dim=-1)

        # Z_r = -1/2 * sum_d ((v_d - m_rd) / sigma_rd)^2, expanded into three
        # matmuls rather than materialising the (B, L, R, D) difference tensor
        # (which is ~1 GB at B=2048, L=64, R=32, D=64).
        inv_var = torch.exp(-2.0 * self.log_sigma)  # (R, D)
        quad = v.pow(2) @ inv_var.t()  # (B, L, R)  sum_d v_d^2 / s_rd^2
        cross = v @ (self.centers * inv_var).t()  # sum_d v_d m_rd / s_rd^2
        const = (self.centers.pow(2) * inv_var).sum(-1)  # (R,)
        z = -0.5 * (quad - 2.0 * cross + const)  # (B, L, R)

        # The product t-norm is never formed explicitly, so it cannot underflow:
        # everything stays in the log domain until the softmax.

        if self.defuzzify == DEFUZZ_LOG:
            # LogTSK: l1-normalise -1/Z. Heavier tailed than softmax, so more
            # rules stay active.
            recip = 1.0 / (-z + 1e-6)
            return recip / recip.sum(-1, keepdim=True).clamp_min(1e-12)

        # HTSK: divide the exponent by D (learned). ``product`` leaves it alone
        # and is expected to saturate -- that is the point of ablation A3.
        if self.defuzzify == DEFUZZ_HTSK:
            z = z / torch.exp(self.log_tau).clamp_min(1e-6)
        return torch.softmax(z, dim=-1)

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        input_ids = features["input_ids"]
        mask = features["attention_mask"].to(self.features.weight.dtype)

        v = self.norm(self.features(input_ids))  # (B, L, d_in)

        # Pooling weights: masked softmax over the learned per-token logit. This
        # is a trainable generalisation of SIF/IDF weighting.
        if self.pool == "learned":
            logit = self.pool_logit(input_ids).squeeze(-1)  # (B, L)
            logit = logit.masked_fill(mask == 0, torch.finfo(logit.dtype).min)
            attn = torch.softmax(logit, dim=-1)
        else:
            attn = mask / mask.sum(-1, keepdim=True).clamp_min(1e-9)
        attn = attn * mask  # guard fully-masked rows

        context = None
        if self.context_conditioned:
            # One extra cheap pass: the document's mean token feature vector.
            # O(L), no token-token interaction -- this is fuzzy gating, not attention.
            context = torch.einsum("bl,bld->bd", attn, v).unsqueeze(1)

        f_bar = self.firing_strengths(v, context)  # (B, L, R)
        if self.training and self.rule_dropout > 0:
            f_bar = self._drop(f_bar)

        # Record diagnostics over real tokens: mean firing (for the UR loss and
        # the rule-usage metric) and mean per-token firing entropy (for the
        # saturation metric). These measure different failures -- see
        # ``rule_entropy`` vs ``firing_entropy``.
        denom = mask.sum().clamp_min(1.0)
        self.last_firing_mean = (f_bar * mask.unsqueeze(-1)).sum((0, 1)) / denom
        # Kept differentiable (not under no_grad) so the optional fuzziness anchor
        # in FuzzyContrastiveLoss can actually backprop through it.
        ent = -(f_bar.clamp_min(1e-12).log() * f_bar).sum(-1)  # (B, L)
        self.last_token_entropy = (ent * mask).sum() / denom

        # Consequent + defuzzification over the sequence, fused.
        #
        # The naive route computes u_t = sum_r f_r(v_t) (A_r v_t + b_r) per token,
        # which needs a (B, L, R, d_out) intermediate -- 4 GB at B=2048. But the
        # sequence pool is a *fixed linear* combination, so it commutes inward:
        #
        #   e_d = sum_l a_l sum_r f_lr ( sum_i v_li A_rid + b_rd )
        #       = sum_r sum_i [ sum_l a_l f_lr v_li ] A_rid  +  sum_r [ sum_l a_l f_lr ] b_rd
        #
        # so define the per-rule pooled feature G_ri and per-rule mass h_r, and the
        # expert projection runs once per *document* instead of once per token.
        # Cost drops from O(L R d_in d_out) to O(L R d_in) + O(R d_in d_out).
        # This is exact, not an approximation.
        aw = attn.unsqueeze(-1) * f_bar  # (B, L, R)
        h = aw.sum(1)  # (B, R)     per-rule mass
        emb = h @ self.expert_b  # (B, d_out)
        if self.consequent_order == 1:
            g = torch.einsum("blr,bli->bri", aw, v)  # (B, R, d_in)
            if self.consequent_rank:
                # Contract to the shared k-dim basis first, so the (k, d_out)
                # matrix is touched once per batch rather than once per rule.
                z = torch.einsum("bri,rik->bk", g, self.expert_u)
                emb = emb + z @ self.shared_v
            else:
                emb = emb + torch.einsum("bri,rid->bd", g, self.expert_w)

        features["sentence_embedding"] = F.normalize(emb, p=2, dim=-1)
        return features

    # ------------------------------------------------------------------ metrics
    def uniform_regularisation(self) -> torch.Tensor:
        """PyTSK's Uniform Regularisation: ``sum_r (mean f_bar_r - 1/R)^2``.

        Penalises rule-usage imbalance. Without it a subset of rules goes unused
        and the effective parameter count silently drops -- this is the fuzzy
        literature's name for the mixture-of-experts load-balancing loss.
        """
        if self.last_firing_mean is None:
            return torch.zeros((), device=self.centers.device)
        tau = 1.0 / self.n_rules
        return (self.last_firing_mean - tau).pow(2).sum()

    def rule_entropy(self) -> float:
        """**Rule-usage** entropy: ``H(mean_t f_bar) / log R``, in ``[0, 1]``.

        Answers "are all rules used *somewhere*?". 1.0 = every rule carries equal
        total load; ~0 = the rule base has collapsed and most experts are dead
        weight, which makes the reported parameter count a fiction. This is what
        the UR loss targets.
        """
        if self.last_firing_mean is None or self.n_rules == 1:
            return float("nan")
        p = self.last_firing_mean.detach().float()
        p = p / p.sum().clamp_min(1e-12)
        h = -(p * torch.log(p.clamp_min(1e-12))).sum()
        return float(h / math.log(self.n_rules))

    def firing_entropy(self) -> float:
        """**Per-token** firing entropy: ``mean_t H(f_bar_t) / log R``, in ``[0, 1]``.

        Answers "is the inference actually fuzzy?". This is the metric that
        detects the arXiv:2102.04271 saturation failure, and it is *not* the same
        as :meth:`rule_entropy`: if every token fires exactly one rule but
        different tokens pick different rules, usage entropy stays near 1.0 while
        this goes to 0. Such a model is a hard router, not a fuzzy system -- it
        loses the smooth interpolation between local experts that motivates TSK
        in the first place, and its antecedent gradients vanish.
        """
        if getattr(self, "last_token_entropy", None) is None or self.n_rules == 1:
            return float("nan")
        return float(self.last_token_entropy.detach() / math.log(self.n_rules))

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_pool_weights_from_frequency(
        self, counts: np.ndarray, sif_coefficient: float = 1e-3
    ) -> None:
        """SIF-initialise the pooling logits: ``w = a / (a + p(t))``.

        Starts the model at the classical strong baseline (POTION's
        re-regularisation step) instead of at uniform pooling.
        """
        p = counts.astype(np.float64)
        p = p / max(p.sum(), 1.0)
        w = sif_coefficient / (sif_coefficient + p)
        self.pool_logit.weight.copy_(
            torch.from_numpy(np.log(np.clip(w, 1e-8, None))).float().unsqueeze(-1)
        )

    @torch.no_grad()
    def calibrate_temperature(
        self, token_ids: np.ndarray, target_entropy: float = 0.5, iters: int = 40
    ) -> float:
        """Set the HTSK temperature so mean per-token firing entropy hits a target.

        **Why this exists.** HTSK (arXiv:2102.04271) fixes the softmax *saturation*
        end of the problem by dividing the exponent by ``D``. But ``D`` is not a
        calibrated choice, and at our scale (LayerNorm'd features, sigma from
        KMeans cluster spread) it over-corrects into the opposite degeneracy:
        ``f_bar -> 1/R`` uniformly. That is just as fatal, because

            u_t = (1/R) sum_r (A_r v_t + b_r) = A_bar v_t + b_bar

        i.e. a *single* linear map -- the rule base has collapsed again, silently,
        while rule-usage entropy reads a perfect 1.0.

        So the rule base has two failure modes at opposite ends of one axis:

        ==================  ==============  ==================  ================
        temperature         firing entropy  effective experts   failure
        ==================  ==============  ==================  ================
        too low (product)   -> 0            1 (hard routing)    saturation, dead grads
        too high (raw HTSK) -> 1            1 (mean of experts) uniform blending
        ==================  ==============  ==================  ================

        Both give one effective expert. The useful regime is strictly in between,
        and neither ``1`` nor ``D`` lands there reliably. This routine picks it by
        bisection on ``log_tau`` against a target normalised entropy, so the
        choice is made against an observable rather than by convention.

        ``target_entropy = 0.5`` implies ``exp(0.5 log R)`` effective rules per
        token -- e.g. ~5.7 of 32. Diffuse enough to keep gradients flowing to
        every rule, sharp enough that rules actually specialise.

        Returns:
            The calibrated ``exp(log_tau)``.
        """
        if self.n_rules == 1 or self.defuzzify != DEFUZZ_HTSK:
            return float(self.log_tau.exp())

        ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(
            self.features.weight.device
        )
        v = self.norm(self.features(ids)).unsqueeze(0)  # (1, N, d_in)
        ctx = v.mean(1, keepdim=True) if self.context_conditioned else None

        def entropy_at(log_tau: float) -> float:
            saved = self.log_tau.detach().clone()
            self.log_tau.data.fill_(log_tau)
            f = self.firing_strengths(v, ctx)
            self.log_tau.data.copy_(saved)
            h = -(f.clamp_min(1e-12).log() * f).sum(-1).mean()
            return float(h / math.log(self.n_rules))

        # Entropy is monotone increasing in log_tau; bracket then bisect.
        lo, hi = -8.0, 12.0
        if entropy_at(lo) > target_entropy:
            self.log_tau.data.fill_(lo)
            return float(math.exp(lo))
        if entropy_at(hi) < target_entropy:
            self.log_tau.data.fill_(hi)
            return float(math.exp(hi))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if entropy_at(mid) < target_entropy:
                lo = mid
            else:
                hi = mid
        self.log_tau.data.fill_(0.5 * (lo + hi))
        # Record the achieved entropy so firing_entropy() reports the calibrated
        # value rather than a stale one from before calibration.
        self.last_token_entropy = torch.tensor(
            entropy_at(float(self.log_tau)) * math.log(self.n_rules)
        )
        return float(self.log_tau.exp())

    @torch.no_grad()
    def init_centers_from_kmeans(self, token_ids: np.ndarray, seed: int = 42) -> None:
        """Place rule centres with KMeans on the current token features.

        Standard TSK scatter-partition initialisation (PyTSK). ``token_ids``
        should be the most frequent vocabulary items -- placing centres where the
        data actually is, not uniformly over the feature space.
        """
        from sklearn.cluster import KMeans

        ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(
            self.features.weight.device
        )
        v = self.norm(self.features(ids))
        if self.context_conditioned:
            v = torch.cat([v, v.mean(0, keepdim=True).expand_as(v)], dim=-1)
        x = v.detach().float().cpu().numpy()
        km = KMeans(n_clusters=self.n_rules, random_state=seed, n_init=4).fit(x)
        self.centers.copy_(torch.from_numpy(km.cluster_centers_).to(self.centers))
        # sigma = h * per-dimension std of the assigned cluster (h = 1); HTSK is
        # insensitive to h for h >= 0.5, so this only needs to be roughly right.
        sig = np.empty_like(km.cluster_centers_)
        for r in range(self.n_rules):
            pts = x[km.labels_ == r]
            sig[r] = pts.std(0) if len(pts) > 1 else x.std(0)
        self.log_sigma.copy_(
            torch.from_numpy(np.log(np.clip(sig, 1e-2, None))).to(self.log_sigma)
        )

    # ------------------------------------------------------------------ ST API
    @property
    def vocab(self) -> dict[str, int]:
        """Vocabulary as a dict.

        A Rust ``tokenizers.Tokenizer`` exposes ``get_vocab()`` but no ``.vocab``
        attribute, and ``mteb.model_meta`` probes for the latter when it tries to
        count embedding parameters. Without this it logs a spurious
        ``Error: 'tokenizers.Tokenizer' object has no attribute 'vocab'`` that
        looks like an evaluation failure but is only metadata introspection.
        """
        return self.tokenizer.get_vocab()

    def get_embedding_dimension(self) -> int:
        return self.d_out

    def get_sentence_embedding_dimension(self) -> int:
        return self.d_out

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        if safe_serialization:
            save_safetensors_file(state, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(state, os.path.join(output_path, "pytorch_model.bin"))
        with open(Path(output_path) / self.config_file_name, "w", encoding="utf-8") as fh:
            json.dump(self.get_config_dict(), fh, indent=2)
        self.tokenizer.save(str(Path(output_path) / "tokenizer.json"))

    @classmethod
    def load(cls, model_name_or_path: str, subfolder: str = "", **kwargs) -> "FuzzyEmbedding":
        hub_kwargs = {
            "subfolder": subfolder,
            "token": kwargs.get("token"),
            "cache_folder": kwargs.get("cache_folder"),
            "revision": kwargs.get("revision"),
            "local_files_only": kwargs.get("local_files_only", False),
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        tokenizer_path = cls.load_file_path(
            model_name_or_path, filename="tokenizer.json", **hub_kwargs
        )
        tokenizer = Tokenizer.from_file(tokenizer_path)
        module = cls(tokenizer, **config)
        weights_path = cls.load_file_path(
            model_name_or_path, filename="model.safetensors", **hub_kwargs
        )
        module.load_state_dict(load_safetensors_file(weights_path))
        return module
