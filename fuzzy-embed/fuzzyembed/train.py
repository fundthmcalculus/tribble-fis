"""Contrastive training for FES.

Recipe follows the from-scratch static-embedding blog post (MNRL + Matryoshka,
huge LR on the sparse tables, one epoch), with three additions the fuzzy layer
needs:

1. **Three optimiser groups.** The vocabulary tables want lr ~0.2; the dense
   antecedent/consequent parameters want ~1e-3; the single HTSK temperature
   scalar wants ~50x the dense LR or it cannot move within one epoch. One LR for
   all three either diverges or crawls.
2. **Uniform Regularisation** on rule usage (PyTSK), so the rule base does not
   collapse to a handful of live experts.
3. **Temperature calibration** against a target per-token firing entropy, run
   after KMeans centre init. Raw HTSK (tau = D) produced near-uniform firing in
   the E001 smoke run, which collapses the mixture of experts to their average --
   a second, quieter degeneracy that rule-usage entropy does not detect.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from torch import nn
from transformers import TrainerCallback

from fuzzyembed.model import FuzzyEmbedding

logger = logging.getLogger(__name__)


@dataclass
class FESConfig:
    """One rung of the ablation ladder in DESIGN.md §5."""

    name: str = "A2-fes-s"
    d_in: int = 64
    d_out: int = 256
    n_rules: int = 32
    consequent_order: int = 1
    context_conditioned: bool = False
    defuzzify: str = "htsk"
    learn_temperature: bool = True
    feature_norm: bool = True
    rule_dropout: float = 0.0
    pool: str = "learned"
    # Factorise A_r = U_r @ V with V shared across rules. Decouples the
    # parameter cost of R from d_out; see results/scaling_cost.json.
    consequent_rank: int | None = None
    max_seq_length: int = 256

    # training
    batch_size: int = 2048
    lr_sparse: float = 0.2
    lr_dense: float = 2e-3
    epochs: float = 1.0
    warmup_ratio: float = 0.05
    ur_weight: float = 1.0
    matryoshka_dims: tuple[int, ...] = (256, 128, 64, 32)
    mnrl_scale: float = 20.0
    seed: int = 42
    # NO_DUPLICATES (used by the static-embedding blog) builds a duplicate index
    # over the whole corpus, which dominated wall-clock at 4.1M pairs -- profiling
    # showed tokenisation at 61k texts/s and the GPU step at 55k samples/s, so the
    # sampler was the only candidate left. At batch 4096 over 4.1M pairs an
    # accidental in-batch duplicate is rare enough not to distort InfoNCE.
    batch_sampler: str = "random"  # "random" | "no_duplicates"
    kmeans_init: bool = True
    sif_init: bool = True
    # Target mean per-token firing entropy used to calibrate the HTSK
    # temperature at init. Neither tau=1 (saturates) nor tau=D (uniform) lands in
    # the useful regime on its own -- see FuzzyEmbedding.calibrate_temperature.
    target_firing_entropy: float = 0.5
    lr_temperature: float | None = None
    # Ablation A8: hold per-token firing entropy near target_firing_entropy
    # instead of letting the contrastive objective sharpen it to a hard router.
    entropy_weight: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class FuzzyContrastiveLoss(nn.Module):
    """Matryoshka(MNRL) plus the fuzzy rule-balance penalty.

    The UR term reads the firing statistics recorded by the last forward pass of
    the FuzzyEmbedding module. Because MNRL runs the module once per column
    (anchor, positive, [negative]), ``last_firing_mean`` reflects the final column
    only -- which is fine: it is an unbiased sample of the same distribution, and
    the term is a regulariser, not the objective.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        fuzzy: FuzzyEmbedding,
        scale: float = 20.0,
        matryoshka_dims: tuple[int, ...] = (256, 128, 64, 32),
        ur_weight: float = 1.0,
        entropy_weight: float = 0.0,
        entropy_target: float = 0.5,
    ) -> None:
        super().__init__()
        base = MultipleNegativesRankingLoss(model, scale=scale)
        dims = [d for d in matryoshka_dims if d <= fuzzy.d_out]
        self.inner = (
            MatryoshkaLoss(model, base, matryoshka_dims=dims) if len(dims) > 1 else base
        )
        self.fuzzy = fuzzy
        self.ur_weight = ur_weight
        self.entropy_weight = entropy_weight
        self.entropy_target = entropy_target
        self.last_parts: dict[str, float] = {}

    def forward(self, sentence_features, labels=None):
        loss = self.inner(sentence_features, labels)
        ur = self.fuzzy.uniform_regularisation()
        # Scale UR by R^2 so its magnitude is comparable across rule counts:
        # a perfectly collapsed base gives sum_r (f_r - 1/R)^2 ~ 1, and a
        # uniformly-perturbed one scales as 1/R. Without this, ur_weight would
        # need retuning for every R and the A6 sweep would be uninterpretable.
        ur_scaled = ur * (self.fuzzy.n_rules**2)
        total = loss + self.ur_weight * ur_scaled

        # Optional fuzziness anchor (ablation A8). Left at 0 by default.
        #
        # E001 showed the contrastive objective drives per-token firing entropy
        # from a calibrated 0.50 down to ~0.04 -- it *prefers* a crisp router over
        # a fuzzy one, while keeping every rule in use. That is a legitimate model
        # (a hard mixture of experts, still a TSK system with narrow sigma) but it
        # abandons the smooth interpolation between local experts that motivates
        # fuzzy inference. This term holds entropy near the target so the two
        # regimes can be compared rather than assumed.
        ent_pen = torch.zeros((), device=total.device)
        if self.entropy_weight > 0 and self.fuzzy.n_rules > 1:
            h = self.fuzzy.last_token_entropy / math.log(self.fuzzy.n_rules)
            ent_pen = (h - self.entropy_target).pow(2)
            total = total + self.entropy_weight * ent_pen

        self.last_parts = {
            "contrastive": float(loss.detach()),
            "ur_raw": float(ur.detach()),
            "entropy_penalty": float(ent_pen.detach()),
            "rule_entropy": self.fuzzy.rule_entropy(),
            "firing_entropy": self.fuzzy.firing_entropy(),
            "log_tau": float(self.fuzzy.log_tau.detach()),
        }
        return total

    def get_config_dict(self) -> dict:
        return {"ur_weight": self.ur_weight}


class FuzzyMetricsCallback(TrainerCallback):
    """Pushes the fuzzy diagnostics (rule entropy, firing entropy, tau) into the
    trainer's log stream, and keeps a local history so the run can be plotted
    without a tracking service."""

    def __init__(self, loss: FuzzyContrastiveLoss) -> None:
        self.loss = loss
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            extra = {k: v for k, v in self.loss.last_parts.items() if v == v}
            logs.update(extra)
            self.history.append({"step": state.global_step, **logs})
        return control


def build_model(cfg: FESConfig, tokenizer) -> tuple[SentenceTransformer, FuzzyEmbedding]:
    fuzzy = FuzzyEmbedding(
        tokenizer,
        d_in=cfg.d_in,
        d_out=cfg.d_out,
        n_rules=cfg.n_rules,
        consequent_order=cfg.consequent_order,
        context_conditioned=cfg.context_conditioned,
        defuzzify=cfg.defuzzify,
        learn_temperature=cfg.learn_temperature,
        feature_norm=cfg.feature_norm,
        rule_dropout=cfg.rule_dropout,
        pool=cfg.pool,
        consequent_rank=cfg.consequent_rank,
        max_seq_length=cfg.max_seq_length,
    )
    model = SentenceTransformer(modules=[fuzzy])
    return model, fuzzy


def make_optimizer(fuzzy: FuzzyEmbedding, cfg: FESConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        fuzzy.param_groups(cfg.lr_sparse, cfg.lr_dense, cfg.lr_temperature)
    )


def train(
    cfg: FESConfig,
    train_mix: dict,
    tokenizer,
    output_dir: str,
    eval_mix: dict | None = None,
    evaluator=None,
    steps_per_log: int = 50,
    eval_steps: int | None = None,
    token_counts=None,
    kmeans_ids=None,
    report_to: str = "none",
) -> tuple[SentenceTransformer, FuzzyEmbedding, dict]:
    torch.manual_seed(cfg.seed)
    model, fuzzy = build_model(cfg, tokenizer)

    if cfg.sif_init and token_counts is not None:
        fuzzy.init_pool_weights_from_frequency(token_counts)
        logger.info("pooling weights SIF-initialised from corpus frequencies")
    if cfg.kmeans_init and kmeans_ids is not None and cfg.n_rules > 1:
        fuzzy.init_centers_from_kmeans(kmeans_ids, seed=cfg.seed)
        logger.info("rule centres KMeans-initialised (R=%d)", cfg.n_rules)
    if kmeans_ids is not None and cfg.n_rules > 1:
        # Must run *after* KMeans, since sigma from cluster spread changes the
        # entropy landscape entirely.
        tau = fuzzy.calibrate_temperature(kmeans_ids, cfg.target_firing_entropy)
        logger.info(
            "HTSK temperature calibrated: tau=%.3f (D=%d), firing entropy target %.2f -> %.4f",
            tau, fuzzy.D, cfg.target_firing_entropy, fuzzy.firing_entropy(),
        )

    counts = fuzzy.parameter_counts()
    logger.info("parameter budget: %s", counts)

    loss = FuzzyContrastiveLoss(
        model, fuzzy,
        scale=cfg.mnrl_scale,
        matryoshka_dims=cfg.matryoshka_dims,
        ur_weight=cfg.ur_weight,
        entropy_weight=cfg.entropy_weight,
        entropy_target=cfg.target_firing_entropy,
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        # Transformers v5 takes the ratio through `warmup_steps` as a float.
        warmup_steps=cfg.warmup_ratio,
        bf16=torch.cuda.is_available(),
        batch_sampler=(
            BatchSamplers.NO_DUPLICATES
            if cfg.batch_sampler == "no_duplicates"
            else BatchSamplers.BATCH_SAMPLER
        ),
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        logging_steps=steps_per_log,
        eval_strategy="steps" if (evaluator or eval_mix) and eval_steps else "no",
        eval_steps=eval_steps,
        save_strategy="no",
        seed=cfg.seed,
        report_to=report_to,
        dataloader_num_workers=0,
        learning_rate=cfg.lr_dense,  # overridden per-group by our optimizer
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_mix,
        eval_dataset=eval_mix,
        loss=loss,
        evaluator=evaluator,
        optimizers=(make_optimizer(fuzzy, cfg), None),
    )
    metrics_cb = FuzzyMetricsCallback(loss)
    trainer.add_callback(metrics_cb)
    result = trainer.train()

    info = {
        "config": cfg.as_dict(),
        "params": counts,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "final_rule_entropy": fuzzy.rule_entropy(),
        "final_firing_entropy": fuzzy.firing_entropy(),
        "final_log_tau": float(fuzzy.log_tau.detach()),
        "log_history": metrics_cb.history,
    }
    model.save(output_dir)
    return model, fuzzy, info
