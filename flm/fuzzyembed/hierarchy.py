"""The named hierarchy that supplies the embedding's dimensions.

Every coordinate of a fuzzy embedding is a membership degree in a *named node* of
this tree. That is the whole point: the axes have names before training starts, so
there is no post-hoc naming problem to solve (see ``../README.md``).

Backend
-------
The plan (``../FUZZY_EMBEDDING_PLAN.md``) specifies Roget's Thesaurus. Roget's is
not reachable from this environment -- Open Roget's is on sites.google.com and the
1911 edition on gutenberg.org, both outside the egress allowlist -- so this module
uses **WordNet**, which *is* reachable via the ``nltk_data`` GitHub mirror.

The substitution is closer than it sounds. WordNet's 45 lexicographer files
(``noun.animal``, ``verb.motion``, ``adj.all``, ...) are supersenses playing exactly
the role of Roget's 39 Sections, and the hypernym chain supplies the depth that
Roget's Head Groups and Heads supply. What is lost is Roget's antonymous opposed
pairs (648 Goodness / 649 Badness), which would have collapsed into signed bipolar
axes for free; WordNet has no such pairing, so polarity has to come from elsewhere
(``senses.py`` uses the opinion lexicon). Swap in Roget's by writing another
``build_*_hierarchy`` returning the same ``FuzzyHierarchy``.

The level ladder
----------------
Each synset gets a canonical root-first path::

    *  ->  pos  ->  lexname  ->  <same-lexname hypernym chain>  ->  synset

    *  ->  n  ->  noun.animal  ->  animal -> chordate -> vertebrate
                                  -> mammal -> placental -> carnivore
                                  -> canine -> dog

Restricting the hypernym chain to ancestors sharing the synset's lexname is what
keeps this a *tree* rather than WordNet's DAG, and keeps each path semantically
coherent (an ancestor of ``dog`` under ``noun.animal`` is always an animal, never
``entity``).

Level ``L`` of a synset is ``path[min(L, len(path) - 1)]`` -- paths shorter than
``L`` clamp to the synset itself. Clamping is what makes the rollup identity in
``rollup()`` hold *exactly* at every level, which is the property that
distinguishes this from Matryoshka truncation. ``tests/test_hierarchy.py``
asserts it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Level indices into the canonical path.
L_ROOT, L_POS, L_LEX = 0, 1, 2

ROOT_KEY = "*"


@dataclass(frozen=True)
class Node:
    """One named dimension of the embedding space."""

    key: str          # stable id, e.g. "wn:dog.n.01", "lex:noun.animal", "pos:n"
    name: str         # human-readable label used in rules and explanations
    path: tuple[str, ...]  # keys root-first, inclusive of self

    @property
    def depth(self) -> int:
        return len(self.path) - 1

    @property
    def parent(self) -> str | None:
        return self.path[-2] if len(self.path) > 1 else None


def _pos_group(pos: str) -> str:
    """Merge WordNet's satellite adjectives ('s') into the adjective class."""
    return "a" if pos == "s" else pos


def _canonical_path(synset, lexname_restricted: bool = True) -> tuple[str, ...]:
    """Root-first key path for a synset (see module docstring).

    ``lexname_restricted=True`` (the default) keeps only same-supersense ancestors in
    the chain. **This is required for correctness, not a stylistic choice.** Exact
    rollup needs node paths to be *prefix-consistent*: every prefix of a terminal's
    path must be the registered path of that ancestor. Dropping the restriction
    breaks it, because an ancestor can sit under a different supersense than its
    descendant -- ``dog.n.01`` is ``lex:noun.animal`` but its ancestor
    ``entity.n.01`` is ``lex:noun.Tops``, so ``entity``'s own canonical path
    ``(*, pos:n, lex:noun.Tops, entity)`` is not a prefix of ``dog``'s. Rollup then
    silently disagrees between levels and the multi-resolution claim is void. It was
    tried, and ``run_flm --stage embed`` reported the exactness check FAIL along with
    scrambled L2 similarities.

    The cost is an unbalanced ladder -- measured widths ``[1, 4, 45, 4526, 7257,
    9862]``, a ~100x jump from L2 to L3. Two causes: same-supersense chains are
    short, and **WordNet has no adjective hypernyms** at all (adjectives are
    organised by antonymy and similarity, not subsumption), so every adjective and
    adverb clamps at depth 3. A uniform ladder over all parts of speech is therefore
    intrinsically unbalanced here. Roget's, a designed 5-level tree covering every
    part of speech with known per-level cardinality, would not have this problem.
    This is the sharpest measured cost of the WordNet substitution.
    """
    lex = synset.lexname()
    pos = _pos_group(synset.pos())
    prefix = (ROOT_KEY, f"pos:{pos}", f"lex:{lex}")

    try:
        hyper_paths = synset.hypernym_paths()
    except Exception:  # noqa: BLE001 - WordNet raises on some malformed entries
        hyper_paths = []
    if not hyper_paths:
        return prefix + (f"wn:{synset.name()}",)

    # Shortest path is the canonical one, which is what makes this a tree rather
    # than WordNet's DAG.
    best = min(hyper_paths, key=len)
    chain = tuple(
        f"wn:{a.name()}" for a in best
        if (not lexname_restricted) or a.lexname() == lex or a.name() == synset.name()
    )
    self_key = f"wn:{synset.name()}"
    if not chain or chain[-1] != self_key:
        chain = chain + (self_key,)
    return prefix + chain


class FuzzyHierarchy:
    """A tree of named nodes, with exact multi-resolution readout."""

    def __init__(self, nodes: dict[str, Node], n_levels: int,
                 terminals: set[str] | None = None):
        """``terminals`` are the keys that senses actually land on (the synsets).

        Levels are derived from the terminals' paths, *not* from every registered
        node. Interior nodes still live in ``nodes`` for naming and parent lookup,
        but they must not generate their own dimensions: because short paths clamp
        (which is what makes the rollup exact), registering an ancestor as a node in
        its own right made it clamp into every deeper level as an extra,
        near-always-on coordinate. The root was the worst case -- it appeared at
        every resolution carrying the max of everything, inflating similarity
        denominators for no information.
        """
        self.nodes = nodes
        self.n_levels = n_levels
        self.terminals = set(terminals) if terminals is not None else set(nodes)
        self._level_keys: dict[int, list[str]] = {}
        self._level_index: dict[int, dict[str, int]] = {}
        self._plans: dict[tuple[int, int], tuple] = {}
        for level in range(n_levels):
            keys = sorted({self.project(k, level) for k in self.terminals})
            self._level_keys[level] = keys
            self._level_index[level] = {k: i for i, k in enumerate(keys)}

    # -- structure ---------------------------------------------------------

    def project(self, key: str, level: int) -> str:
        """The ancestor of ``key`` at ``level``, clamping to ``key`` if shallower."""
        path = self.nodes[key].path
        return path[min(level, len(path) - 1)]

    def level_keys(self, level: int) -> list[str]:
        return self._level_keys[level]

    def width(self, level: int) -> int:
        return len(self._level_keys[level])

    def index(self, key: str, level: int) -> int:
        return self._level_index[level][key]

    def name(self, key: str) -> str:
        node = self.nodes.get(key)
        return node.name if node else key

    def leaf_keys(self) -> list[str]:
        return self._level_keys[self.n_levels - 1]

    def widths(self) -> list[int]:
        return [self.width(level) for level in range(self.n_levels)]

    # -- readout -----------------------------------------------------------

    def zeros(self, level: int) -> np.ndarray:
        return np.zeros(self.width(level), dtype=np.float32)

    def _rollup_plan(self, from_level: int, to_level: int):
        """Cached grouping that turns a rollup into one vectorised reduction.

        ``rollup`` used to walk every key at ``from_level`` and call ``project`` per key,
        per call. That is a pure-Python loop over up to 9,864 nodes, repeated for every
        level of every token, and it dominated the whole pipeline: profiling attributed
        **8.7 million ``project`` calls to 400 token types** (21,700 each), and cold
        featurisation of a 3,000-type vocabulary cost 58s, which was 88% of total fit time
        (``../LOG.md`` E23).

        The parent of each coordinate is fixed by the hierarchy, so the mapping
        ``i -> j`` is structural and can be computed once. Sorting the source indices by
        destination makes each destination's sources contiguous, so the aggregation becomes
        a single ``ufunc.reduceat`` -- no Python-level iteration at all.
        """
        cached = self._plans.get((from_level, to_level))
        if cached is not None:
            return cached

        idx = np.empty(self.width(from_level), dtype=np.intp)
        for key, i in self._level_index[from_level].items():
            idx[i] = self._level_index[to_level][self.project(key, to_level)]
        order = np.argsort(idx, kind="stable")
        dest = idx[order]
        # First position of each run of equal destinations.
        starts = np.concatenate(([0], np.nonzero(np.diff(dest))[0] + 1))
        plan = (order, dest[starts], starts)
        self._plans[(from_level, to_level)] = plan
        return plan

    def rollup(self, vec: np.ndarray, from_level: int, to_level: int,
               op: str = "max") -> np.ndarray:
        """Aggregate a level-``from_level`` vector up to ``to_level``.

        This is the exactness claim: a coarse readout is a *named disjunction* of
        its children, not a truncation whose discarded coordinates are unexplained.
        ``op="max"`` is the standard t-conorm; ``"probor"`` is the probabilistic
        sum ``a + b - ab``; ``"sum"`` matches the Ruspini sibling-partition variant
        (and may exceed 1 if the partition is not enforced).

        Vectorised; ``_rollup_reference`` is the original elementwise implementation, kept
        because a test asserts the two agree. The exactness of the multi-resolution readout
        is the central claim of this representation, so the fast path is only allowed to
        stand while it is provably identical to the obvious one.
        """
        if to_level > from_level:
            raise ValueError("rollup only aggregates upward (to_level <= from_level)")
        if op not in ("max", "sum", "probor"):
            raise ValueError(f"unknown rollup op {op!r}")
        if to_level == from_level:
            return vec.astype(np.float32, copy=True)

        order, groups, starts = self._rollup_plan(from_level, to_level)
        vals = np.asarray(vec, dtype=np.float32)[order]
        out = self.zeros(to_level)
        if op == "max":
            out[groups] = np.maximum.reduceat(vals, starts)
        elif op == "sum":
            out[groups] = np.add.reduceat(vals, starts)
        else:
            # probor is 1 - prod(1 - v) over the group: associative and commutative, so
            # a product reduction gives exactly the pairwise a + b - ab accumulation.
            out[groups] = 1.0 - np.multiply.reduceat(1.0 - vals, starts)
        return out

    def _rollup_reference(self, vec: np.ndarray, from_level: int, to_level: int,
                          op: str = "max") -> np.ndarray:
        """The original elementwise rollup. Kept only as a correctness oracle."""
        if to_level > from_level:
            raise ValueError("rollup only aggregates upward (to_level <= from_level)")
        if to_level == from_level:
            return vec.astype(np.float32, copy=True)
        out = self.zeros(to_level)
        for key, i in self._level_index[from_level].items():
            j = self._level_index[to_level][self.project(key, to_level)]
            if op == "max":
                out[j] = max(out[j], vec[i])
            elif op == "sum":
                out[j] += vec[i]
            elif op == "probor":
                out[j] = out[j] + vec[i] - out[j] * vec[i]
            else:
                raise ValueError(f"unknown rollup op {op!r}")
        return out

    def enforce_subsumption(self, vecs: dict[int, np.ndarray],
                            op: str = "max") -> dict[int, np.ndarray]:
        """Rebuild every coarse level from the finest, so (C1) holds by construction.

        Cheaper and more reliable than penalising violations: derive the coarse
        levels instead of learning them.
        """
        finest = self.n_levels - 1
        out = {finest: vecs[finest].astype(np.float32, copy=True)}
        for level in range(finest - 1, -1, -1):
            out[level] = self.rollup(out[level + 1], level + 1, level, op=op)
        return out

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "n_levels": self.n_levels,
            "terminals": sorted(self.terminals),
            "nodes": [
                {"key": n.key, "name": n.name, "path": list(n.path)}
                for n in self.nodes.values()
            ],
        }))

    @classmethod
    def load(cls, path: Path) -> FuzzyHierarchy:
        blob = json.loads(Path(path).read_text())
        nodes = {
            d["key"]: Node(d["key"], d["name"], tuple(d["path"]))
            for d in blob["nodes"]
        }
        return cls(nodes, blob["n_levels"], set(blob.get("terminals") or nodes))

    def describe(self) -> str:
        lines = [f"FuzzyHierarchy: {len(self.nodes)} nodes "
                 f"({len(self.terminals)} terminal), {self.n_levels} levels"]
        for level in range(self.n_levels):
            keys = self.level_keys(level)
            sample = ", ".join(self.name(k) for k in keys[:6])
            lines.append(f"  L{level}: width={len(keys):>6}  e.g. {sample}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# WordNet backend
# --------------------------------------------------------------------------

def wordnet_synsets(lemma: str, wn) -> list:
    """Synsets for a surface lemma, using morphy for inflection stripping."""
    out = wn.synsets(lemma)
    if out:
        return out
    for pos in ("n", "v", "a", "r"):
        base = wn.morphy(lemma, pos)
        if base:
            out = wn.synsets(base, pos=pos)
            if out:
                return out
    return []


def build_wordnet_hierarchy(vocabulary: list[str], n_levels: int = 6,
                            wn=None, lexname_restricted: bool = True
                            ) -> tuple[FuzzyHierarchy, dict[str, list]]:
    """Build a hierarchy covering only the senses ``vocabulary`` can reach.

    Restricting to the corpus vocabulary is what keeps this a *small* model:
    WordNet has 117k synsets, but a children's-story vocabulary touches a small
    fraction, and every unreachable node would be a permanently-zero dimension.

    Returns ``(hierarchy, lemma_to_synsets)``.
    """
    if wn is None:
        from nltk.corpus import wordnet as wn  # noqa: PLC0415

    lemma_synsets: dict[str, list] = {}
    nodes: dict[str, Node] = {ROOT_KEY: Node(ROOT_KEY, "*", (ROOT_KEY,))}
    terminals: set[str] = set()

    for lemma in vocabulary:
        synsets = wordnet_synsets(lemma, wn)
        if not synsets:
            continue
        lemma_synsets[lemma] = synsets
        for syn in synsets:
            path = _canonical_path(syn, lexname_restricted)
            terminals.add(path[-1])
            # Register every ancestor, so interior nodes are real dimensions.
            for depth in range(len(path)):
                key = path[depth]
                if key in nodes:
                    continue
                nodes[key] = Node(key, _pretty(key), path[: depth + 1])

    return FuzzyHierarchy(nodes, n_levels, terminals), lemma_synsets


def _pretty(key: str) -> str:
    """Human-readable label. These strings end up inside fuzzy rules."""
    if key == ROOT_KEY:
        return "*"
    kind, _, rest = key.partition(":")
    if kind == "pos":
        return {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}.get(rest, rest)
    if kind == "lex":
        return rest
    # "dog.n.01" -> "dog"
    return rest.split(".")[0].replace("_", " ")
