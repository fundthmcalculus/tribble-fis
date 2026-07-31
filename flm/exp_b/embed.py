"""Encode a corpus with a frozen neural embedding model and cache the result.

The encoder is frozen and run exactly once. Every head then trains in seconds off
the cache, which is what makes a thorough head sweep affordable.

Matryoshka truncation is exploited deliberately: EmbeddingGemma is trained with
MRL, so a 768-d embedding can be sliced to 512/256/128 with no retraining. The
dimension sweep is therefore free, and it answers the question a FIS designer
actually cares about -- how many embedding dimensions does the rule base need
before the curse of dimensionality bites? See ``../FIS_ON_EMBEDDINGS_PLAN.md``.

Usage
-----
    uv run python flm/exp_b/embed.py --dataset sst2 \
        --model google/embeddinggemma-300m --dims 768 512 256 128 \
        --out flm/exp_b/cache

Needs network on first run (model weights + dataset).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Fallback chain. gte-small needs no auth and is already used by
# tests/test_textclassifier.py, so it is the zero-setup option.
MODELS = {
    "gemma": "google/embeddinggemma-300m",
    "gte-small": "thenlper/gte-small",
    "granite": "ibm-granite/granite-embedding-small-english-r2",
}


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-normalize. Required *after* MRL truncation, not before.

    A truncated prefix of a normalized vector is not itself normalized, and
    feeding un-normalized rows to a distance-based head silently makes the
    effective feature scale depend on the truncation width.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def encode(texts: list[str], model_name: str, batch_size: int = 64,
           max_dim: int | None = None) -> np.ndarray:
    """Encode ``texts`` at full width (or ``max_dim`` if the model supports it)."""
    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": True}
    if max_dim is not None:
        kwargs["truncate_dim"] = max_dim
    model = SentenceTransformer(model_name, **kwargs)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=True),
        dtype=np.float32,
    )


def truncate(X: np.ndarray, dim: int) -> np.ndarray:
    """MRL prefix slice to ``dim``, then re-normalize."""
    if dim > X.shape[1]:
        raise ValueError(f"cannot truncate width {X.shape[1]} to {dim}")
    return l2_normalize(X[:, :dim])


def cache_path(out_dir: Path, dataset: str, model_key: str, dim: int, noise: float) -> Path:
    tag = f"noise{noise:g}" if noise > 0 else "clean"
    return out_dir / f"{dataset}__{model_key}__d{dim}__{tag}.npz"


def build_cache(dataset: str, model_name: str, dims: list[int], out_dir: Path,
                noise_rates: list[float], max_train: int | None,
                max_test: int | None, batch_size: int) -> None:
    from data import load_dataset
    from perturb import perturb_corpus

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset, max_train=max_train, max_test=max_test)
    print(ds.summary())

    model_key = model_name.split("/")[-1]
    full_dim = max(dims)

    for noise in noise_rates:
        # Only the *test* side is perturbed: the deployment story is a model
        # trained on clean text meeting typos in the wild. Perturbing train too
        # would measure noise-augmented training, a different experiment.
        test_texts = (
            ds.test_texts if noise == 0
            else perturb_corpus(ds.test_texts, noise, seed=1234)
        )
        print(f"\n=== encoding (noise={noise:g}) with {model_name} ===")
        train_emb = encode(ds.train_texts, model_name, batch_size, full_dim) \
            if noise == 0 else None
        test_emb = encode(test_texts, model_name, batch_size, full_dim)

        for dim in sorted(dims, reverse=True):
            path = cache_path(out_dir, dataset, model_key, dim, noise)
            payload = {
                "test_X": truncate(test_emb, dim),
                "test_y": ds.test_y,
                "task": ds.task,
            }
            if train_emb is not None:
                payload["train_X"] = truncate(train_emb, dim)
                payload["train_y"] = ds.train_y
            else:
                # Reuse the clean training embeddings; they do not depend on the
                # test-side noise rate.
                clean = np.load(cache_path(out_dir, dataset, model_key, dim, 0.0),
                                allow_pickle=True)
                payload["train_X"] = clean["train_X"]
                payload["train_y"] = clean["train_y"]
            np.savez_compressed(path, **payload)
            print(f"  wrote {path.name}  train={payload['train_X'].shape} "
                  f"test={payload['test_X'].shape}")

    meta = {
        "dataset": dataset, "model": model_name, "dims": dims,
        "noise_rates": noise_rates, "task": ds.task,
        "n_train": len(ds.train_texts), "n_test": len(ds.test_texts),
    }
    (out_dir / f"{dataset}__{model_key}__meta.json").write_text(json.dumps(meta, indent=2))
    # Texts are kept so atlas.py can name dimensions by top-activating documents.
    np.savez_compressed(
        out_dir / f"{dataset}__texts.npz",
        train_texts=np.array(ds.train_texts, dtype=object),
        test_texts=np.array(ds.test_texts, dtype=object),
        allow_pickle=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="sst2", help="sst2 | sst5 | sst_cont | imdb")
    p.add_argument("--model", default=MODELS["gte-small"],
                   help=f"HF model id, or a key of {list(MODELS)}")
    p.add_argument("--dims", type=int, nargs="+", default=[384],
                   help="MRL widths to cache; the largest is the encode width")
    p.add_argument("--noise", type=float, nargs="+", default=[0.0],
                   help="per-token perturbation rates for the test split")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "cache")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    build_cache(args.dataset, MODELS.get(args.model, args.model), args.dims,
                args.out, sorted(set(args.noise)), args.max_train, args.max_test,
                args.batch_size)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
