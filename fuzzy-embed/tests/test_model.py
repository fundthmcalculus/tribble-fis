"""Correctness tests for the FES forward pass.

The important one is ``test_fused_pool_matches_naive``: the forward pass pushes the
sequence pooling inside the consequent to avoid a (B, L, R, d_out) intermediate,
and that refactor is only valid because the pool is a fixed linear combination.
If it ever stops matching the literal per-token definition, the model is wrong.
"""

from __future__ import annotations

import math

import numpy as np

import pytest
import torch
from tokenizers import Tokenizer

from fuzzyembed.model import DEFUZZ_HTSK, DEFUZZ_LOG, DEFUZZ_PROD, FuzzyEmbedding


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("google-bert/bert-base-uncased")


def _make(tokenizer, **kw) -> FuzzyEmbedding:
    torch.manual_seed(0)
    defaults = dict(d_in=16, d_out=24, n_rules=6, max_seq_length=32)
    defaults.update(kw)
    return FuzzyEmbedding(tokenizer, **defaults)


def _naive_forward(m: FuzzyEmbedding, feats: dict) -> torch.Tensor:
    """The literal definition: pool per-token consequent outputs."""
    ids, mask = feats["input_ids"], feats["attention_mask"].float()
    v = m.norm(m.features(ids))
    logit = m.pool_logit(ids).squeeze(-1).masked_fill(mask == 0, torch.finfo(torch.float32).min)
    attn = torch.softmax(logit, -1) * mask
    ctx = torch.einsum("bl,bld->bd", attn, v).unsqueeze(1) if m.context_conditioned else None
    f_bar = m.firing_strengths(v, ctx)
    u = torch.einsum("blr,rd->bld", f_bar, m.expert_b)
    if m.consequent_order == 1:
        a = m.expert_w if m.expert_w is not None else (m.expert_u @ m.shared_v)
        u = u + torch.einsum("blr,rid,bli->bld", f_bar, a, v)
    e = torch.einsum("bl,bld->bd", attn, u)
    return torch.nn.functional.normalize(e, p=2, dim=-1)


@pytest.mark.parametrize("rank", [None, 8])
@pytest.mark.parametrize("context_conditioned", [False, True])
@pytest.mark.parametrize("order", [0, 1])
def test_fused_pool_matches_naive(tokenizer, context_conditioned, order, rank):
    m = _make(tokenizer, context_conditioned=context_conditioned, consequent_order=order,
              consequent_rank=rank).eval()
    feats = m.preprocess(["the quick brown fox", "a much longer sentence about pandas eating"])
    with torch.no_grad():
        fused = m(dict(feats))["sentence_embedding"]
        naive = _naive_forward(m, feats)
    torch.testing.assert_close(fused, naive, rtol=1e-4, atol=1e-6)


def test_firing_strengths_normalised(tokenizer):
    for defuzz in (DEFUZZ_HTSK, DEFUZZ_PROD, DEFUZZ_LOG):
        m = _make(tokenizer, defuzzify=defuzz).eval()
        feats = m.preprocess(["hello world", "fuzzy inference systems"])
        v = m.norm(m.features(feats["input_ids"]))
        f = m.firing_strengths(v)
        assert f.shape[-1] == m.n_rules
        torch.testing.assert_close(f.sum(-1), torch.ones_like(f.sum(-1)), rtol=1e-5, atol=1e-5)
        assert (f >= 0).all()


def test_embeddings_are_unit_norm(tokenizer):
    m = _make(tokenizer).eval()
    feats = m.preprocess(["one", "two three", "four five six seven"])
    with torch.no_grad():
        e = m(feats)["sentence_embedding"]
    torch.testing.assert_close(e.norm(dim=-1), torch.ones(3), rtol=1e-5, atol=1e-5)


def test_padding_does_not_change_embedding(tokenizer):
    """A short sentence must embed identically whether or not it is batched with
    a long one. If the mask is wrong this is the test that catches it."""
    m = _make(tokenizer).eval()
    short = "the quick brown fox"
    with torch.no_grad():
        alone = m(m.preprocess([short]))["sentence_embedding"][0]
        batched = m(m.preprocess([short, "a" + " very" * 20 + " long tail here"]))[
            "sentence_embedding"
        ][0]
    torch.testing.assert_close(alone, batched, rtol=1e-4, atol=1e-6)


def test_htsk_prevents_softmax_saturation(tokenizer):
    """The load-bearing claim from arXiv:2102.04271.

    With the vanilla product t-norm the *per-token* firing distribution must
    collapse to one-hot as D grows; with HTSK it must stay diffuse. This is the
    empirical form of ablation A3.

    Note it is ``firing_entropy`` (per-token) and not ``rule_entropy`` (usage)
    that detects this: a saturated model is still a perfectly *balanced* hard
    router, so usage entropy stays high while the fuzziness is entirely gone.
    """
    ents = {}
    for defuzz in (DEFUZZ_PROD, DEFUZZ_HTSK):
        m = _make(tokenizer, d_in=256, n_rules=16, defuzzify=defuzz).eval()
        feats = m.preprocess(["the quick brown fox jumps over the lazy dog"] * 4)
        with torch.no_grad():
            m(feats)
        ents[defuzz] = m.firing_entropy()
    # Observed at random init, d_in=256, R=16: product 0.14, HTSK 1.00.
    # 0.14 normalised entropy is ~1.5 effective rules out of 16.
    assert ents[DEFUZZ_PROD] < 0.25, f"product t-norm should saturate, got {ents}"
    assert ents[DEFUZZ_HTSK] > 0.8, f"HTSK should stay diffuse, got {ents}"
    assert ents[DEFUZZ_HTSK] > 4 * ents[DEFUZZ_PROD], f"HTSK must dominate, got {ents}"


def test_usage_entropy_and_firing_entropy_are_different_metrics(tokenizer):
    """Guards the distinction above: a saturated product-t-norm model can look
    healthy on rule usage while being a hard router per token."""
    m = _make(tokenizer, d_in=256, n_rules=16, defuzzify=DEFUZZ_PROD).eval()
    with torch.no_grad():
        m(m.preprocess(["alpha beta gamma delta epsilon zeta eta theta iota kappa"]))
    assert m.rule_entropy() > 3 * m.firing_entropy()


def test_single_rule_reduces_to_static_model(tokenizer):
    """R=1 must be exactly a low-rank static embedding model (ablation A1):
    f_bar == 1, so e = normalize(pool(v) @ A_0 + b_0)."""
    m = _make(tokenizer, n_rules=1).eval()
    feats = m.preprocess(["fuzzy logic", "static embeddings"])
    ids, mask = feats["input_ids"], feats["attention_mask"]
    with torch.no_grad():
        v = m.norm(m.features(ids))
        f = m.firing_strengths(v)
        torch.testing.assert_close(f, torch.ones_like(f))

        logit = m.pool_logit(ids).squeeze(-1).masked_fill(mask == 0, torch.finfo(torch.float32).min)
        attn = torch.softmax(logit, -1) * mask
        pooled = torch.einsum("bl,bld->bd", attn, v)
        expected = torch.nn.functional.normalize(
            pooled @ m.expert_w[0] + attn.sum(-1, keepdim=True) * m.expert_b[0], p=2, dim=-1
        )
        torch.testing.assert_close(m(feats)["sentence_embedding"], expected, rtol=1e-4, atol=1e-6)


def test_parameter_counts_match_reality(tokenizer):
    m = _make(tokenizer, d_in=64, d_out=256, n_rules=32)
    assert m.parameter_counts()["total"] == sum(p.numel() for p in m.parameters())


def test_htsk_temperature_initialised_to_D(tokenizer):
    m = _make(tokenizer, d_in=64, context_conditioned=False)
    assert math.isclose(m.log_tau.exp().item(), 64.0, rel_tol=1e-5)
    mc = _make(tokenizer, d_in=64, context_conditioned=True)
    assert math.isclose(mc.log_tau.exp().item(), 128.0, rel_tol=1e-5)


def test_context_conditioning_changes_token_routing(tokenizer):
    """FES-C's whole purpose: the same token must route differently depending on
    the document it appears in. If this fails, FES-C is silently FES-S."""
    m = _make(tokenizer, context_conditioned=True).eval()
    tok = "bank"
    with torch.no_grad():
        a = m.preprocess([f"{tok} river water sediment erosion"])
        b = m.preprocess([f"{tok} loan mortgage interest credit"])
        va, vb = m.norm(m.features(a["input_ids"])), m.norm(m.features(b["input_ids"]))
        ca = va.mean(1, keepdim=True)
        cb = vb.mean(1, keepdim=True)
        fa = m.firing_strengths(va[:, :1], ca)[0, 0]
        fb = m.firing_strengths(vb[:, :1], cb)[0, 0]
    assert not torch.allclose(fa, fb, atol=1e-3)


def test_uniform_firing_collapses_to_a_single_linear_map(tokenizer):
    """The second degeneracy, found in the E001 smoke run.

    If f_bar == 1/R for every token then u_t = mean_r (A_r v_t + b_r), i.e. one
    linear map -- the rule base is gone. Rule-usage entropy reads a perfect 1.0
    throughout, which is why this failure is easy to miss.
    """
    m = _make(tokenizer, n_rules=8, d_in=16, d_out=24).eval()
    m.log_tau.data.fill_(20.0)  # enormous temperature => uniform firing
    feats = m.preprocess(["fuzzy inference over tokens", "another sentence entirely"])
    with torch.no_grad():
        got = m(feats)["sentence_embedding"]
        assert m.firing_entropy() > 0.999

        # Equivalent single-expert model built from the averaged parameters.
        ids, mask = feats["input_ids"], feats["attention_mask"]
        v = m.norm(m.features(ids))
        logit = m.pool_logit(ids).squeeze(-1).masked_fill(mask == 0, torch.finfo(torch.float32).min)
        attn = torch.softmax(logit, -1) * mask
        pooled = torch.einsum("bl,bld->bd", attn, v)
        expected = torch.nn.functional.normalize(
            pooled @ m.expert_w.mean(0) + attn.sum(-1, keepdim=True) * m.expert_b.mean(0),
            p=2, dim=-1,
        )
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-4)


def test_calibrate_temperature_hits_target_entropy(tokenizer):
    """Calibration must land on the requested entropy, for both variants and
    across targets -- this is what replaces the arbitrary tau = D."""
    ids = np.arange(1000, 5000)
    for ctx in (False, True):
        for target in (0.2, 0.5, 0.8):
            m = _make(tokenizer, d_in=32, n_rules=16, context_conditioned=ctx).eval()
            m.init_centers_from_kmeans(ids, seed=0)
            m.calibrate_temperature(ids, target_entropy=target)
            assert abs(m.firing_entropy() - target) < 0.02, (
                f"ctx={ctx} target={target} got={m.firing_entropy():.4f}"
            )


def test_raw_htsk_is_not_automatically_well_calibrated(tokenizer):
    """Documents *why* calibration exists: tau = D leaves firing near-uniform at
    this scale, so HTSK alone does not produce a working rule base."""
    ids = np.arange(1000, 5000)
    m = _make(tokenizer, d_in=32, n_rules=16).eval()
    m.init_centers_from_kmeans(ids, seed=0)
    with torch.no_grad():
        m(m.preprocess(["the quick brown fox jumps over the lazy dog"]))
    raw = m.firing_entropy()
    m.calibrate_temperature(ids, target_entropy=0.5)
    assert raw > 0.9, f"expected raw HTSK to be near-uniform, got {raw}"
    assert m.firing_entropy() < raw


def test_temperature_gets_its_own_param_group(tokenizer):
    m = _make(tokenizer)
    groups = m.param_groups(0.2, 2e-3)
    names = [g["name"] for g in groups]
    assert names == ["sparse_tables", "dense_fis", "htsk_temperature"]
    assert groups[-1]["lr"] > groups[1]["lr"]
    # Every trainable parameter must land in exactly one group.
    assigned = sum(len(g["params"]) for g in groups)
    assert assigned == len([p for p in m.parameters() if p.requires_grad])


def test_save_and_load_roundtrip(tokenizer, tmp_path):
    m = _make(tokenizer, n_rules=5, context_conditioned=True).eval()
    feats = m.preprocess(["roundtrip test sentence"])
    with torch.no_grad():
        before = m(feats)["sentence_embedding"]
    m.save(str(tmp_path))
    m2 = FuzzyEmbedding.load(str(tmp_path)).eval()
    with torch.no_grad():
        after = m2(m2.preprocess(["roundtrip test sentence"]))["sentence_embedding"]
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


def test_uniform_regularisation_is_zero_at_perfect_balance(tokenizer):
    m = _make(tokenizer, n_rules=8)
    m.last_firing_mean = torch.full((8,), 1 / 8)
    assert m.uniform_regularisation().item() == pytest.approx(0.0, abs=1e-12)
    m.last_firing_mean = torch.tensor([1.0] + [0.0] * 7)
    assert m.uniform_regularisation().item() > 0.5


def test_lowrank_consequent_is_equivalent_to_its_dense_expansion(tokenizer):
    """A_r = U_r @ V must behave exactly like a dense A_r built from that product.
    This is what licenses using rank to scale R -- the factorisation restricts the
    expert maps to a shared k-dim output basis but changes nothing else."""
    m = _make(tokenizer, n_rules=6, d_in=16, d_out=24, consequent_rank=8).eval()
    dense = _make(tokenizer, n_rules=6, d_in=16, d_out=24).eval()
    dense.load_state_dict(
        {k: v for k, v in m.state_dict().items() if k not in ("expert_u", "shared_v")},
        strict=False,
    )
    with torch.no_grad():
        dense.expert_w.copy_(m.expert_u @ m.shared_v)
        feats = m.preprocess(["low rank consequents", "another test sentence here"])
        a = m(dict(feats))["sentence_embedding"]
        b = dense(dict(feats))["sentence_embedding"]
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_lowrank_cuts_parameters_at_large_R(tokenizer):
    """The scaling claim: rank makes R cheap. Dense R=512 costs >10M; rank-32 <3.5M."""
    dense = _make(tokenizer, d_in=64, d_out=256, n_rules=512)
    low = _make(tokenizer, d_in=64, d_out=256, n_rules=512, consequent_rank=32)
    assert dense.parameter_counts()["total"] > 10_000_000
    assert low.parameter_counts()["total"] < 3_500_000
    for mod in (dense, low):
        assert mod.parameter_counts()["total"] == sum(p.numel() for p in mod.parameters())


def test_lowrank_survives_save_load(tokenizer, tmp_path):
    m = _make(tokenizer, n_rules=8, consequent_rank=4).eval()
    with torch.no_grad():
        before = m(m.preprocess(["roundtrip low rank"]))["sentence_embedding"]
    m.save(str(tmp_path))
    m2 = FuzzyEmbedding.load(str(tmp_path)).eval()
    assert m2.consequent_rank == 4
    with torch.no_grad():
        after = m2(m2.preprocess(["roundtrip low rank"]))["sentence_embedding"]
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


def test_lowrank_caps_the_embedding_rank(tokenizer):
    """The failure E012 missed: `V` is SHARED across rules, so the model's entire
    output is confined to a k-dimensional subspace (plus the bias term's rank).

    `test_lowrank_consequent_is_equivalent_to_its_dense_expansion` passes for a
    rank-limited model because that expansion is *itself* rank-limited -- it checks
    the algebra, not the modelling consequence. This test checks the consequence:
    a rank-k model cannot fill its own d_out.
    """
    k, R, d_out = 8, 6, 64
    m = _make(tokenizer, d_in=16, d_out=d_out, n_rules=R, consequent_rank=k).eval()
    texts = [f"sentence number {i} covering assorted unrelated subject matter" for i in range(200)]
    with torch.no_grad():
        e = m(m.preprocess(texts))["sentence_embedding"].numpy()
    sv = np.linalg.svd(e - e.mean(0), compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    assert rank <= k + R, f"rank {rank} exceeds the k+R={k + R} bound"
    assert rank < d_out, "a rank-limited model should not fill d_out"

    dense = _make(tokenizer, d_in=16, d_out=d_out, n_rules=R).eval()
    with torch.no_grad():
        ed = dense(dense.preprocess(texts))["sentence_embedding"].numpy()
    svd_ = np.linalg.svd(ed - ed.mean(0), compute_uv=False)
    assert int((svd_ > svd_[0] * 1e-6).sum()) == d_out, "dense should be full rank"


def test_lowrank_warns_when_rank_would_bottleneck_output(tokenizer):
    """Using consequent_rank far below d_out is almost always a mistake; the model
    must say so rather than silently producing padded low-rank vectors."""
    with pytest.warns(UserWarning, match="embedding rank"):
        _make(tokenizer, d_in=64, d_out=512, n_rules=32, consequent_rank=32)
