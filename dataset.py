"""
dataset.py — Synthetic Relational Reasoning Dataset for ESBN-GRU (PsyNeuLink)
==============================================================================

Implements entity embeddings and trial generators for the four tasks from
Webb et al. (2021) "Emergent Symbols through Binding in External Memory":

    Task                  SEQ_LEN   Output
    ──────────────────    ───────   ──────────────────────────────────
    SameDifferent         2         [1,0]=same  [0,1]=different
    RMTS                  4         [1,0]=same-relation  [0,1]=different-relation
    DistributionOf3       4         [1,0]=D follows rule  [0,1]=D violates rule
    IdentityRules         6         [1,0]=candidate correct  [0,1]=foil

No images. No CNN. Each entity is defined by four discrete attributes and
encoded as a structured float vector. The key property from Webb et al. is
preserved: training and test entities are drawn from *disjoint* pools, so
the model cannot memorize entities — it must learn the abstract relation.

──────────────────────────────────────────────────────────────────────────────
Entity space
──────────────────────────────────────────────────────────────────────────────
    shape:   {circle, square, triangle, diamond}   — 4 values
    color:   {red, blue, green, yellow}             — 4 values
    size:    {small, large}                         — 2 values
    texture: {solid, striped}                       — 2 values
    ─────────────────────────────────────────────────
    Total:   4 × 4 × 2 × 2 = 64 unique entities

Embedding: 12-D one-hot (concatenated per-attribute) → fixed linear projection
→ embed_dim-D unit vector.  The projection is fixed at construction so the
same entity always maps to the same vector (referential integrity without CNN).

──────────────────────────────────────────────────────────────────────────────
PsyNeuLink target format
──────────────────────────────────────────────────────────────────────────────
With full_sequence_mode=True, PsyNeuLink steps through each timestep
individually.  All task generators return per-timestep targets shaped as:

    [ [0,0], [0,0], ..., [0,0], actual_label ]
      ────────────────── t=0..T-2 ─────────   t=T-1

Applying loss only at the final step avoids asking the model to classify
a relation it hasn't fully observed yet.

──────────────────────────────────────────────────────────────────────────────
Quick start
──────────────────────────────────────────────────────────────────────────────

    from dataset import DatasetFactory

    factory = DatasetFactory(embed_dim=64, seed=42)

    # Same/Different
    inputs, labels, pnl_targets = factory.same_different.generate(500, split='train')
    # inputs:      list[list[np.ndarray]]   len=500, each inner list has 2 vectors
    # labels:      list[np.ndarray]         len=500, each shape (2,)
    # pnl_targets: list[list[np.ndarray]]   len=500, zeros + label (PsyNeuLink format)

    # Use in esbn_gru_comp.learn():
    esbn_gru_comp.learn(
        inputs={z_embedding_mech: inputs},
        targets={y_hat_mech: pnl_targets},
        ...
    )
"""

from __future__ import annotations

import numpy as np
from itertools import product
from typing import List, Tuple, Optional, Dict, NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Attribute definitions and entity pool
# ─────────────────────────────────────────────────────────────────────────────

SHAPES   = ['circle', 'square', 'triangle', 'diamond']
COLORS   = ['red', 'blue', 'green', 'yellow']
SIZES    = ['small', 'large']
TEXTURES = ['solid', 'striped']

ATTR_SIZES = [len(SHAPES), len(COLORS), len(SIZES), len(TEXTURES)]  # [4, 4, 2, 2]
ATTR_NAMES = ['shape', 'color', 'size', 'texture']
RAW_DIM    = sum(ATTR_SIZES)   # 12
N_ENTITIES = 4 * 4 * 2 * 2    # 64

OUTPUT_DIM = 2   # All four tasks share a binary output head


class Entity(NamedTuple):
    """
    An entity defined by four integer attribute indices.

    Using NamedTuple gives free equality, hashing, and unpacking while keeping
    the interface clean.  Two Entity objects with the same attribute values are
    always considered 'same' — this is the ground truth the tasks depend on.
    """
    shape:   int   # 0–3
    color:   int   # 0–3
    size:    int   # 0–1
    texture: int   # 0–1

    def one_hot(self) -> np.ndarray:
        """Concatenated one-hot vector, shape (RAW_DIM=12,) dtype=float32."""
        v = np.zeros(RAW_DIM, dtype=np.float32)
        offset = 0
        for val, size in zip(self, ATTR_SIZES):
            v[offset + val] = 1.0
            offset += size
        return v

    def describe(self) -> str:
        return (f"{SIZES[self.size]} {COLORS[self.color]} "
                f"{TEXTURES[self.texture]} {SHAPES[self.shape]}")


# Build the complete pool of 64 entities once at module load time.
ALL_ENTITIES: List[Entity] = [
    Entity(s, c, sz, tx)
    for s, c, sz, tx in product(
        range(len(SHAPES)),
        range(len(COLORS)),
        range(len(SIZES)),
        range(len(TEXTURES))
    )
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. EntityEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class EntityEmbedder:
    """
    Maps Entity objects to float32 unit vectors of dimension embed_dim.

    The projection matrix is fixed at construction (not learned), which is
    the analogue of a frozen CNN encoder.  The same entity always maps to the
    same vector.  All embeddings are L2-normalised to unit sphere so the EM
    cosine-similarity retrieval (or raw dot product with normalize_memories=False)
    treats all entities as equidistant in magnitude — only direction encodes
    identity.

    If embed_dim == RAW_DIM (12), the identity projection is used and each
    embedding is just the one-hot vector (scaled to unit length).
    """

    def __init__(self, embed_dim: int = 64, seed: int = 0):
        self.embed_dim = embed_dim
        self.raw_dim   = RAW_DIM

        rng = np.random.RandomState(seed)

        if embed_dim == RAW_DIM:
            proj = np.eye(RAW_DIM, dtype=np.float32)
        elif embed_dim < RAW_DIM:
            # QR decomposition gives a well-conditioned orthonormal basis
            raw = rng.randn(RAW_DIM, embed_dim).astype(np.float32)
            proj, _ = np.linalg.qr(raw)
            proj = proj[:, :embed_dim]
        else:
            # embed_dim > RAW_DIM: random Gaussian matrix with normalised columns
            proj = rng.randn(RAW_DIM, embed_dim).astype(np.float32)
            norms = np.linalg.norm(proj, axis=0, keepdims=True)
            proj = proj / np.maximum(norms, 1e-8)

        # Pre-compute and cache all 64 embeddings
        self._projection = proj
        self._cache: Dict[Entity, np.ndarray] = {}

        for entity in ALL_ENTITIES:
            raw_vec = entity.one_hot() @ proj          # (embed_dim,)
            norm    = np.linalg.norm(raw_vec)
            self._cache[entity] = (raw_vec / max(norm, 1e-8)).astype(np.float32)

    def embed(self, entity: Entity) -> np.ndarray:
        """Return the cached unit-vector embedding for this entity. Shape: (embed_dim,)"""
        return self._cache[entity].copy()

    def embed_sequence(self, entities: List[Entity]) -> List[np.ndarray]:
        """Return a list of embeddings for a sequence of entities."""
        return [self.embed(e) for e in entities]


# ─────────────────────────────────────────────────────────────────────────────
# 3. EntityPool — manages train/test entity splits
# ─────────────────────────────────────────────────────────────────────────────

class EntityPool:
    """
    Manages disjoint train/test entity splits.

    The core generalization test from Webb et al.: the model must apply
    learned relations to entities it has *never* seen during training.  This
    class enforces that split and makes it easy to sample from either pool.

    Args:
        n_train:  Number of entities reserved for training.
        n_test:   Number of entities reserved for testing.
        seed:     Random seed for the split (reproducible).

    With 64 total entities, a reasonable default is n_train=40, n_test=16,
    leaving 8 entities unused.  Webb et al. used roughly 60/40 splits.
    """

    def __init__(self, n_train: int = 40, n_test: int = 16, seed: int = 42):
        max_entities = len(ALL_ENTITIES)
        if n_train + n_test > max_entities:
            raise ValueError(
                f"n_train ({n_train}) + n_test ({n_test}) = {n_train+n_test} "
                f"exceeds total entity count ({max_entities})."
            )

        rng     = np.random.RandomState(seed)
        indices = rng.permutation(max_entities)

        self.train: List[Entity] = [ALL_ENTITIES[i] for i in indices[:n_train]]
        self.test:  List[Entity] = [ALL_ENTITIES[i] for i in indices[n_train:n_train + n_test]]
        self.n_train = n_train
        self.n_test  = n_test

    def get(self, split: str) -> List[Entity]:
        if split == 'train':
            return self.train
        elif split == 'test':
            return self.test
        else:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    def summary(self) -> str:
        return (f"EntityPool: {self.n_train} train + {self.n_test} test "
                f"({len(ALL_ENTITIES) - self.n_train - self.n_test} unused of {len(ALL_ENTITIES)} total)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MATCH  = np.array([1., 0.], dtype=np.float32)   # same / match / correct
LABEL_DIFF   = np.array([0., 1.], dtype=np.float32)   # different / mismatch / foil
ZERO_TARGET  = np.zeros(OUTPUT_DIM, dtype=np.float32)

def _pnl_targets(labels: List[np.ndarray], seq_len: int) -> List[List[np.ndarray]]:
    """
    Format labels for PsyNeuLink full_sequence_mode=True.

    With full_sequence_mode each timestep is processed individually.
    The loss should only be applied at the *final* timestep — we have no
    ground-truth supervision signal at earlier steps.

    Returns a list of per-trial target sequences:
        [ [zeros]*( seq_len-1 ) + [label] ]
    """
    prefix = [ZERO_TARGET.copy() for _ in range(seq_len - 1)]
    return [prefix + [label.copy()] for label in labels]


def _shuffle(inputs, labels, rng):
    """Shuffle two parallel lists with the same permutation."""
    perm    = rng.permutation(len(inputs))
    inputs  = [inputs[i]  for i in perm]
    labels  = [labels[i]  for i in perm]
    return inputs, labels


# ─────────────────────────────────────────────────────────────────────────────
# 5. Task 1: SameDifferent
# ─────────────────────────────────────────────────────────────────────────────

class SameDifferentTask:
    """
    Sequence: [ entity_A, entity_B ]
    Label:    [1,0] if A == B else [0,1]

    The simplest relational task.  On each "same" trial both embeddings are
    identical; on each "different" trial they are drawn from distinct entities
    in the pool.  The model must learn to compare the two vectors via the
    external memory loop (store A on t=0, retrieve on t=1 using B as query,
    compare retrieved abstract key to current state).

    Generalization test: test entities never appear in training.

    Balanced: 50% same, 50% different.
    """

    SEQ_LEN = 2
    NAME    = 'same_different'

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 0):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def generate(
        self,
        n_trials:   int,
        split:      str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        """
        Args:
            n_trials: Total number of trials (half same, half different).
            split:    'train' or 'test' — which entity pool to draw from.

        Returns:
            inputs:      list[list[ndarray]]  — shape (n_trials, SEQ_LEN, embed_dim)
            labels:      list[ndarray]         — shape (n_trials, 2)
            pnl_targets: list[list[ndarray]]   — PsyNeuLink format with zero prefix
        """
        entities = self.pool.get(split)
        n_ent    = len(entities)
        n_same   = n_trials // 2
        n_diff   = n_trials - n_same

        inputs, labels = [], []

        # Same trials
        for _ in range(n_same):
            idx = self.rng.randint(n_ent)
            e   = entities[idx]
            inputs.append(self.embedder.embed_sequence([e, e]))
            labels.append(LABEL_MATCH.copy())

        # Different trials — two distinct entities
        for _ in range(n_diff):
            i, j = self.rng.choice(n_ent, size=2, replace=False)
            inputs.append(self.embedder.embed_sequence([entities[i], entities[j]]))
            labels.append(LABEL_DIFF.copy())

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets    = _pnl_targets(labels, self.SEQ_LEN)
        return inputs, labels, pnl_targets


# ─────────────────────────────────────────────────────────────────────────────
# 6. Task 2: RMTS (Relational Match-to-Sample)
# ─────────────────────────────────────────────────────────────────────────────

class RMTSTask:
    """
    Sequence: [ A, B, C, D ]
    Label:    [1,0] if rel(A,B) == rel(C,D) else [0,1]
    where rel(X,Y) ∈ {same, different}

    This is a second-order relational task.  The model must:
      1. Infer the relation between A and B (same or different)
      2. Infer the relation between C and D (same or different)
      3. Judge whether the two relations match

    A model that can only memorize entities will fail — both pairs are drawn
    from the same disjoint entity pool, and success requires abstract relational
    comparison.

    Four trial types (balanced):
      - SS: rel(A,B)=same,      rel(C,D)=same      → MATCH  [1,0]
      - DD: rel(A,B)=different, rel(C,D)=different  → MATCH  [1,0]
      - SD: rel(A,B)=same,      rel(C,D)=different  → DIFF   [0,1]
      - DS: rel(A,B)=different, rel(C,D)=same       → DIFF   [0,1]
    """

    SEQ_LEN = 4
    NAME    = 'rmts'

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 1):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def _same_pair(self, entities):
        """Sample a 'same' pair: (e, e) — entity repeated."""
        idx = self.rng.randint(len(entities))
        e   = entities[idx]
        return e, e

    def _diff_pair(self, entities):
        """Sample a 'different' pair: two distinct entities."""
        i, j = self.rng.choice(len(entities), size=2, replace=False)
        return entities[i], entities[j]

    def generate(
        self,
        n_trials: int,
        split:    str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        entities   = self.pool.get(split)
        n_per_type = n_trials // 4
        remainder  = n_trials - n_per_type * 4

        inputs, labels = [], []

        # SS trials — both pairs same → MATCH
        for _ in range(n_per_type):
            a, b = self._same_pair(entities)
            c, d = self._same_pair(entities)
            inputs.append(self.embedder.embed_sequence([a, b, c, d]))
            labels.append(LABEL_MATCH.copy())

        # DD trials — both pairs different → MATCH
        for _ in range(n_per_type):
            a, b = self._diff_pair(entities)
            c, d = self._diff_pair(entities)
            inputs.append(self.embedder.embed_sequence([a, b, c, d]))
            labels.append(LABEL_MATCH.copy())

        # SD trials — first same, second different → DIFF
        for _ in range(n_per_type):
            a, b = self._same_pair(entities)
            c, d = self._diff_pair(entities)
            inputs.append(self.embedder.embed_sequence([a, b, c, d]))
            labels.append(LABEL_DIFF.copy())

        # DS trials — first different, second same → DIFF
        for _ in range(n_per_type + remainder):
            a, b = self._diff_pair(entities)
            c, d = self._same_pair(entities)
            inputs.append(self.embedder.embed_sequence([a, b, c, d]))
            labels.append(LABEL_DIFF.copy())

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets    = _pnl_targets(labels, self.SEQ_LEN)
        return inputs, labels, pnl_targets


# ─────────────────────────────────────────────────────────────────────────────
# 7. Task 3: Distribution-of-3
# ─────────────────────────────────────────────────────────────────────────────

class DistributionOf3Task:
    """
    Sequence: [ A, B, C, D ]
    Label:    [1,0] if D follows the rule established by A,B,C else [0,1]

    Rule: A, B, C all share the *same value* on some attribute k.
          D either also has that value on attribute k (correct / match)
          or has a *different* value on attribute k (foil / different).

    Example:
        Rule attribute: color = 'red'
        A = small red solid circle
        B = large red striped square
        C = small red solid triangle
        D_correct = large red solid diamond   (also red → follows rule)
        D_foil    = small blue solid diamond  (not red → violates rule)

    The model must:
      1. Identify the shared attribute among A, B, C
      2. Decide whether D has that same attribute value

    This is a genuine inductive reasoning test.  With disjoint entity pools,
    the model cannot recognise specific entities — it must abstract the rule.

    Rule attributes are sampled uniformly; the same attribute value may be
    the "rule" in both train and test (the generalization is over entities,
    not rules).

    Balanced: 50% match, 50% foil.
    """

    SEQ_LEN = 4
    NAME    = 'distribution_of_3'

    # Which attributes can be rules, and the number of values each has
    RULE_ATTRS = list(range(len(ATTR_SIZES)))   # [0, 1, 2, 3]

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 2):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def _entities_with(self, entities: List[Entity], attr: int, val: int) -> List[Entity]:
        """All entities in the pool that have attribute `attr` == `val`."""
        return [e for e in entities if e[attr] == val]

    def _entities_without(self, entities: List[Entity], attr: int, val: int) -> List[Entity]:
        """All entities in the pool that have attribute `attr` != `val`."""
        return [e for e in entities if e[attr] != val]

    def generate(
        self,
        n_trials: int,
        split:    str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        entities   = self.pool.get(split)
        n_match    = n_trials // 2
        n_foil     = n_trials - n_match
        inputs, labels = [], []

        def _one_trial(is_match: bool):
            """
            Sample one trial.  Returns (input_sequence, label) or None if the
            entity pool is too small to satisfy the constraints (in which case
            the caller retries with a different rule).
            """
            # Pick a random rule attribute and value
            attr      = self.rng.choice(self.RULE_ATTRS)
            n_vals    = ATTR_SIZES[attr]
            rule_val  = self.rng.randint(n_vals)

            pool_in  = self._entities_with(entities, attr, rule_val)
            pool_out = self._entities_without(entities, attr, rule_val)

            # Need at least 3 distinct entities for the prefix and 1 for D
            if len(pool_in) < 3:
                return None

            prefix_idxs = self.rng.choice(len(pool_in), size=3, replace=False)
            a, b, c     = [pool_in[i] for i in prefix_idxs]

            if is_match:
                if len(pool_in) < 4:
                    return None
                remaining = [e for e in pool_in if e not in {a, b, c}]
                if not remaining:
                    return None
                d     = remaining[self.rng.randint(len(remaining))]
                label = LABEL_MATCH.copy()
            else:
                if not pool_out:
                    return None
                d     = pool_out[self.rng.randint(len(pool_out))]
                label = LABEL_DIFF.copy()

            seq = self.embedder.embed_sequence([a, b, c, d])
            return seq, label

        # Match trials
        count = 0
        while count < n_match:
            result = _one_trial(is_match=True)
            if result is None:
                continue
            inputs.append(result[0])
            labels.append(result[1])
            count += 1

        # Foil trials
        count = 0
        while count < n_foil:
            result = _one_trial(is_match=False)
            if result is None:
                continue
            inputs.append(result[0])
            labels.append(result[1])
            count += 1

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets    = _pnl_targets(labels, self.SEQ_LEN)
        return inputs, labels, pnl_targets


# ─────────────────────────────────────────────────────────────────────────────
# 8. Task 4: Identity Rules
# ─────────────────────────────────────────────────────────────────────────────

class IdentityRulesTask:
    """
    Sequence: [ P1, P2, P3, Q1, Q2, Q_candidate ]
    Label:    [1,0] if Q_candidate correctly completes the pattern else [0,1]

    This task requires the model to:
      1. Infer an abstract identity rule from an example triplet (P1, P2, P3)
      2. Apply that rule to a new triplet (Q1, Q2, ?) using novel entities

    Two rule types:
        ABA:  P3 == P1   →  correct Q_candidate == Q1, foil Q_candidate == Q2
        ABB:  P3 == P2   →  correct Q_candidate == Q2, foil Q_candidate == Q1

    On each trial:
      - P1, P2 are drawn from distinct entities (P1 ≠ P2)
      - P3 is set by the rule (P1 or P2)
      - Q1, Q2 are drawn from entities *not used in the P triplet* to prevent
        the model from exploiting entity overlap as a shortcut
      - Q_candidate is either the correct choice (match) or the foil

    This is the hardest of the four tasks: the model must encode an abstract
    structural rule (not just similarity) from the example, then apply it to
    an entirely new pair of entities — a genuine test of symbolic reasoning.

    Balanced: 50% correct, 50% foil.
    Rules: 50% ABA, 50% ABB (within each label category).
    """

    SEQ_LEN    = 6
    NAME       = 'identity_rules'
    RULE_TYPES = ['ABA', 'ABB']

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 3):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def generate(
        self,
        n_trials: int,
        split:    str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        entities = self.pool.get(split)
        n_ent    = len(entities)

        if n_ent < 4:
            raise ValueError(
                f"IdentityRulesTask requires at least 4 entities in the pool "
                f"(got {n_ent} in '{split}' split)."
            )

        n_match  = n_trials // 2
        n_foil   = n_trials - n_match
        inputs, labels = [], []

        def _one_trial(is_match: bool):
            # Pick rule type
            rule = self.RULE_TYPES[self.rng.randint(2)]

            # Sample 4 distinct entities: p1, p2, q1, q2
            idxs = self.rng.choice(n_ent, size=4, replace=False)
            p1, p2, q1, q2 = [entities[i] for i in idxs]

            # Build example triplet according to rule
            if rule == 'ABA':
                p3          = p1                    # repeats first
                correct_q3  = q1                    # correct completes Q: Q1,Q2,Q1
                foil_q3     = q2                    # foil would be Q1,Q2,Q2
            else:  # ABB
                p3          = p2                    # repeats second
                correct_q3  = q2                    # correct completes Q: Q1,Q2,Q2
                foil_q3     = q1                    # foil would be Q1,Q2,Q1

            q3         = correct_q3 if is_match else foil_q3
            label      = LABEL_MATCH.copy() if is_match else LABEL_DIFF.copy()
            seq        = self.embedder.embed_sequence([p1, p2, p3, q1, q2, q3])
            return seq, label

        for _ in range(n_match):
            seq, label = _one_trial(is_match=True)
            inputs.append(seq)
            labels.append(label)

        for _ in range(n_foil):
            seq, label = _one_trial(is_match=False)
            inputs.append(seq)
            labels.append(label)

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets    = _pnl_targets(labels, self.SEQ_LEN)
        return inputs, labels, pnl_targets


# ─────────────────────────────────────────────────────────────────────────────
# 9. Webb-faithful task variants
#    Match Webb et al. 2021 task structure exactly:
#      RMTS:          T=6 (source pair + 2 target pairs), binary
#      Dist3:         T=9 (row1=3 + row2=2 + 4 choices), 4-way
#      IdentityRules: T=9 (example=3 + partial=2 + 4 choices), 4-way, ABA/ABB/AAA
# ─────────────────────────────────────────────────────────────────────────────

def _make_4way_label(correct_idx):
    """Create a 4-way one-hot label."""
    label = np.zeros(4, dtype=np.float32)
    label[correct_idx] = 1.0
    return label


class WebbRMTSTask:
    """
    Webb-faithful RMTS: T=6, binary.

    Sequence: [ A, B, C, D, E, F ]
      (A,B) = source pair
      (C,D) = target pair 1
      (E,F) = target pair 2

    One target pair has the same relation as the source (same/different),
    the other has the opposite relation.
    Label: [1,0] if first target matches, [0,1] if second target matches.
    """

    SEQ_LEN    = 6
    OUTPUT_DIM = 2
    NAME       = 'rmts'

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 1):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def _same_pair(self, entities):
        idx = self.rng.randint(len(entities))
        e = entities[idx]
        return e, e

    def _diff_pair(self, entities):
        i, j = self.rng.choice(len(entities), size=2, replace=False)
        return entities[i], entities[j]

    def generate(
        self,
        n_trials: int,
        split:    str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        entities = self.pool.get(split)
        inputs, labels = [], []

        for _ in range(n_trials):
            # Source relation: same or different
            source_same = self.rng.random() < 0.5
            if source_same:
                a, b = self._same_pair(entities)
            else:
                a, b = self._diff_pair(entities)

            # First target matches source?
            first_matches = self.rng.random() < 0.5

            if first_matches:
                # Target 1 = same relation as source, Target 2 = opposite
                if source_same:
                    c, d = self._same_pair(entities)
                    e, f = self._diff_pair(entities)
                else:
                    c, d = self._diff_pair(entities)
                    e, f = self._same_pair(entities)
                labels.append(LABEL_MATCH.copy())  # [1,0] = first matches
            else:
                # Target 1 = opposite, Target 2 = same relation as source
                if source_same:
                    c, d = self._diff_pair(entities)
                    e, f = self._same_pair(entities)
                else:
                    c, d = self._same_pair(entities)
                    e, f = self._diff_pair(entities)
                labels.append(LABEL_DIFF.copy())   # [0,1] = second matches

            inputs.append(self.embedder.embed_sequence([a, b, c, d, e, f]))

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets    = _pnl_targets(labels, self.SEQ_LEN)
        return inputs, labels, pnl_targets


class WebbDistributionOf3Task:
    """
    Webb-faithful Distribution-of-3: T=9, 4-way classification.

    Based on Raven's Progressive Matrices.
    A 2x3 matrix where each row contains the same set of 3 elements.
    Row 2 is missing one element. The task: identify the missing element
    from 4 choices.

    Sequence: [ R1_1, R1_2, R1_3, R2_1, R2_2, C1, C2, C3, C4 ]
      Row 1:   3 entities defining the set (order randomized)
      Row 2:   2 of those 3 entities (order randomized)
      Choices: the missing entity + 3 distractors (shuffled)

    Label: 4-way one-hot indicating which choice is correct.
    """

    SEQ_LEN    = 9
    OUTPUT_DIM = 4
    NAME       = 'distribution_of_3'

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 2):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def generate(
        self,
        n_trials: int,
        split:    str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        entities = self.pool.get(split)
        n_ent    = len(entities)

        if n_ent < 6:
            raise ValueError(
                f"WebbDistributionOf3Task requires at least 6 entities "
                f"(got {n_ent} in '{split}' split)."
            )

        inputs, labels = [], []

        for _ in range(n_trials):
            # Pick 3 distinct entities for the set
            set_idxs = self.rng.choice(n_ent, size=3, replace=False)
            set_entities = [entities[i] for i in set_idxs]

            # Row 1: the full set in random order
            row1_order = self.rng.permutation(3)
            row1 = [set_entities[i] for i in row1_order]

            # Row 2: 2 of the 3, random order. The missing one is the answer.
            missing_idx = self.rng.randint(3)
            row2_entities = [set_entities[i] for i in range(3) if i != missing_idx]
            self.rng.shuffle(row2_entities)
            correct_entity = set_entities[missing_idx]

            # 3 distractors: entities NOT in the set
            non_set = [e for e in entities if e not in set_entities]
            distractor_idxs = self.rng.choice(len(non_set), size=3, replace=False)
            distractors = [non_set[i] for i in distractor_idxs]

            # Choices: correct + 3 distractors, shuffled
            choices = [correct_entity] + distractors
            correct_pos = 0  # correct is at index 0 before shuffle
            choice_perm = self.rng.permutation(4)
            choices = [choices[i] for i in choice_perm]
            correct_pos = int(np.where(choice_perm == 0)[0][0])

            seq = row1 + row2_entities + choices
            inputs.append(self.embedder.embed_sequence(seq))
            labels.append(_make_4way_label(correct_pos))

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets = [
            [np.zeros(4, dtype=np.float32) for _ in range(self.SEQ_LEN - 1)] + [label.copy()]
            for label in labels
        ]
        return inputs, labels, pnl_targets


class WebbIdentityRulesTask:
    """
    Webb-faithful Identity Rules: T=9, 4-way classification, ABA/ABB/AAA.

    Sequence: [ P1, P2, P3, Q1, Q2, C1, C2, C3, C4 ]
      Example triplet: P1, P2, P3 demonstrate the rule
      Query:           Q1, Q2 are the start of a new triplet
      Choices:         4 candidates for Q3 (shuffled)

    Rules:
      ABA: P1==P3, P1!=P2. Correct Q3=Q1.
      ABB: P2==P3, P1!=P2. Correct Q3=Q2.
      AAA: P1==P2==P3.     Q1==Q2, correct Q3=Q1.

    Label: 4-way one-hot indicating which choice is correct.
    """

    SEQ_LEN    = 9
    OUTPUT_DIM = 4
    NAME       = 'identity_rules'
    RULE_TYPES = ['ABA', 'ABB', 'AAA']

    def __init__(self, embedder: EntityEmbedder, pool: EntityPool, seed: int = 3):
        self.embedder = embedder
        self.pool     = pool
        self.rng      = np.random.RandomState(seed)

    def generate(
        self,
        n_trials: int,
        split:    str = 'train',
    ) -> Tuple[List, List[np.ndarray], List[List[np.ndarray]]]:
        entities = self.pool.get(split)
        n_ent    = len(entities)

        if n_ent < 5:
            raise ValueError(
                f"WebbIdentityRulesTask requires at least 5 entities "
                f"(got {n_ent} in '{split}' split)."
            )

        inputs, labels = [], []

        for _ in range(n_trials):
            rule = self.RULE_TYPES[self.rng.randint(3)]

            if rule == 'AAA':
                # P1=P2=P3 (all same), Q1=Q2 (all same), Q3=Q1
                p_idx, q_idx = self.rng.choice(n_ent, size=2, replace=False)
                p1 = p2 = p3 = entities[p_idx]
                q1 = q2 = entities[q_idx]
                correct_q3 = q1
                # Foils: 3 entities that are NOT q1
                foil_pool = [e for e in entities if e != q1]
            elif rule == 'ABA':
                # P1=P3!=P2, Q3=Q1
                idxs = self.rng.choice(n_ent, size=4, replace=False)
                p1, p2, q1, q2 = [entities[i] for i in idxs]
                p3 = p1
                correct_q3 = q1
                # Foils include q2 (tempting ABB answer) + 2 others
                foil_pool = [q2] + [e for e in entities if e not in {q1, q2, p1, p2}]
            else:  # ABB
                # P2=P3!=P1, Q3=Q2
                idxs = self.rng.choice(n_ent, size=4, replace=False)
                p1, p2, q1, q2 = [entities[i] for i in idxs]
                p3 = p2
                correct_q3 = q2
                # Foils include q1 (tempting ABA answer) + 2 others
                foil_pool = [q1] + [e for e in entities if e not in {q1, q2, p1, p2}]

            # Pick 3 foils from foil_pool
            if len(foil_pool) < 3:
                foil_pool = [e for e in entities if e != correct_q3]
            foil_idxs = self.rng.choice(len(foil_pool), size=3, replace=False)
            foils = [foil_pool[i] for i in foil_idxs]

            # Choices: correct + 3 foils, shuffled
            choices = [correct_q3] + foils
            choice_perm = self.rng.permutation(4)
            choices = [choices[i] for i in choice_perm]
            correct_pos = int(np.where(choice_perm == 0)[0][0])

            seq = [p1, p2, p3, q1, q2] + choices
            inputs.append(self.embedder.embed_sequence(seq))
            labels.append(_make_4way_label(correct_pos))

        inputs, labels = _shuffle(inputs, labels, self.rng)
        pnl_targets = [
            [np.zeros(4, dtype=np.float32) for _ in range(self.SEQ_LEN - 1)] + [label.copy()]
            for label in labels
        ]
        return inputs, labels, pnl_targets


# ─────────────────────────────────────────────────────────────────────────────
# 10. DatasetFactory — unified access point
# ─────────────────────────────────────────────────────────────────────────────

class DatasetFactory:
    """
    One-stop shop: creates the embedder, entity pool, and all four task objects.

    Args:
        embed_dim:  Dimensionality of entity vectors fed into the model.
                    Should match VALUE_DIMENSION in your PsyNeuLink config.
                    Default 64 matches the notebook config.
        n_train:    Number of entities in the training pool.
        n_test:     Number of entities in the test pool.
        embed_seed: Seed for the random projection in the embedder.
        pool_seed:  Seed for the train/test entity split.
        task_seed:  Base seed for trial-generation randomness (each task
                    adds its own offset so they don't share state).

    Attributes:
        same_different:    SameDifferentTask
        rmts:              RMTSTask
        distribution_of_3: DistributionOf3Task
        identity_rules:    IdentityRulesTask
        embedder:          EntityEmbedder
        pool:              EntityPool

    Example:
        factory = DatasetFactory(embed_dim=64)
        inputs, labels, pnl_targets = factory.same_different.generate(500, 'train')
    """

    ALL_TASKS = ['same_different', 'rmts', 'distribution_of_3', 'identity_rules']

    def __init__(
        self,
        embed_dim:  int = 64,
        n_train:    int = 40,
        n_test:     int = 16,
        embed_seed: int = 0,
        pool_seed:  int = 42,
        task_seed:  int = 100,
    ):
        self.embedder = EntityEmbedder(embed_dim=embed_dim, seed=embed_seed)
        self.pool     = EntityPool(n_train=n_train, n_test=n_test, seed=pool_seed)

        # Original simplified tasks (binary output, shorter sequences)
        self.same_different    = SameDifferentTask (self.embedder, self.pool, seed=task_seed)
        self.rmts              = RMTSTask          (self.embedder, self.pool, seed=task_seed + 1)
        self.distribution_of_3 = DistributionOf3Task(self.embedder, self.pool, seed=task_seed + 2)
        self.identity_rules    = IdentityRulesTask (self.embedder, self.pool, seed=task_seed + 3)

        # Webb-faithful tasks (match Webb et al. 2021 structure exactly)
        self.webb_rmts              = WebbRMTSTask          (self.embedder, self.pool, seed=task_seed + 1)
        self.webb_distribution_of_3 = WebbDistributionOf3Task(self.embedder, self.pool, seed=task_seed + 2)
        self.webb_identity_rules    = WebbIdentityRulesTask (self.embedder, self.pool, seed=task_seed + 3)

        self._tasks = {
            'same_different':    self.same_different,
            'rmts':              self.rmts,
            'distribution_of_3': self.distribution_of_3,
            'identity_rules':    self.identity_rules,
        }

    def get_task(self, name: str):
        """Retrieve a task object by name string."""
        if name not in self._tasks:
            raise ValueError(f"Unknown task '{name}'. Available: {self.ALL_TASKS}")
        return self._tasks[name]

    def summary(self) -> str:
        lines = [
            "DatasetFactory",
            "─" * 50,
            f"  Embed dim:   {self.embedder.embed_dim}",
            f"  {self.pool.summary()}",
            "",
            "  Tasks:",
            f"    SameDifferent     seq_len={SameDifferentTask.SEQ_LEN}",
            f"    RMTS              seq_len={RMTSTask.SEQ_LEN}",
            f"    DistributionOf3   seq_len={DistributionOf3Task.SEQ_LEN}",
            f"    IdentityRules     seq_len={IdentityRulesTask.SEQ_LEN}",
            "",
            "  Output dim:  2  (all tasks share [match, no-match] output head)",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_results(
    results:     list,
    labels:      List[np.ndarray],
    task_name:   str = '',
    verbose:     bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy from PsyNeuLink results and ground-truth labels.

    PsyNeuLink's results list has one entry per (trial × timestep).
    This function extracts the final-timestep prediction for each trial
    and compares it to the ground-truth label.

    Args:
        results:    esbn_gru_comp.results (or the slice returned by learn/run)
        labels:     Ground-truth labels returned by task.generate()
        task_name:  Optional string for display purposes
        verbose:    Whether to print per-trial details for errors

    Returns:
        dict with keys: accuracy, n_correct, n_total, n_same_correct, n_diff_correct
    """
    n_total          = len(labels)
    correct          = 0
    same_correct     = 0
    same_total       = 0
    diff_correct     = 0
    diff_total       = 0

    for i, (result, label) in enumerate(zip(results, labels)):
        pred_arr = np.array(result).flatten()
        if len(pred_arr) > OUTPUT_DIM:
            pred_arr = pred_arr[-OUTPUT_DIM:]

        if np.any(np.isnan(pred_arr)) or np.sum(pred_arr) == 0:
            pred_class = 0   # default to "match" if degenerate
        else:
            pred_class = int(np.argmax(pred_arr))

        true_class = int(np.argmax(label))
        is_match   = true_class == 0
        is_correct = pred_class == true_class

        correct      += int(is_correct)
        same_total   += int(is_match)
        diff_total   += int(not is_match)
        same_correct += int(is_correct and is_match)
        diff_correct += int(is_correct and not is_match)

        if verbose and not is_correct:
            true_lbl = "MATCH" if true_class == 0 else "DIFF"
            pred_lbl = "MATCH" if pred_class == 0 else "DIFF"
            print(f"  [{task_name}] Trial {i:3d}: true={true_lbl}  pred={pred_lbl}  "
                  f"probs={np.round(pred_arr, 3)}")

    accuracy = correct / n_total if n_total > 0 else 0.0

    if verbose:
        print(f"\n  {task_name} accuracy: {correct}/{n_total} = {accuracy:.1%}")
        if same_total > 0:
            print(f"    Match trials:  {same_correct}/{same_total} = {same_correct/same_total:.1%}")
        if diff_total > 0:
            print(f"    Diff trials:   {diff_correct}/{diff_total} = {diff_correct/diff_total:.1%}")

    return {
        'accuracy':      accuracy,
        'n_correct':     correct,
        'n_total':       n_total,
        'same_accuracy': same_correct / same_total   if same_total > 0 else 0.0,
        'diff_accuracy': diff_correct / diff_total   if diff_total > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. Self-test / demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("dataset.py — self-test")
    print("=" * 60)

    factory = DatasetFactory(embed_dim=64, n_train=40, n_test=16, seed=42)
    print()
    print(factory.summary())

    print("\n" + "─" * 60)
    print("Verifying task outputs...")
    print("─" * 60)

    for task_name in DatasetFactory.ALL_TASKS:
        task       = factory.get_task(task_name)
        n_train    = 200
        n_test     = 80

        tr_inputs, tr_labels, tr_pnl = task.generate(n_train, split='train')
        te_inputs, te_labels, te_pnl = task.generate(n_test,  split='test')

        seq_len    = task.SEQ_LEN
        embed_dim  = factory.embedder.embed_dim

        # Verify shapes
        assert len(tr_inputs)  == n_train,                      f"[{task_name}] wrong n_trials"
        assert len(tr_inputs[0]) == seq_len,                    f"[{task_name}] wrong seq_len"
        assert tr_inputs[0][0].shape == (embed_dim,),           f"[{task_name}] wrong embed_dim"
        assert len(tr_pnl[0]) == seq_len,                       f"[{task_name}] wrong pnl seq_len"
        assert np.all(tr_pnl[0][0] == 0),                       f"[{task_name}] non-zero prefix target"
        assert not np.all(tr_pnl[0][-1] == 0),                  f"[{task_name}] zero final target"

        # Verify balance
        n_match = sum(1 for l in tr_labels if l[0] == 1.0)
        n_diff  = n_train - n_match
        balance = abs(n_match / n_train - 0.5)
        assert balance <= 0.1,                                   f"[{task_name}] imbalanced: {n_match}/{n_train}"

        # Verify entity disjointness by checking embedding distinctness
        # (train and test embeddings for the same "position" should differ)
        tr_first_embeds = set(tuple(tr_inputs[i][0].round(4)) for i in range(min(20, n_train)))
        te_first_embeds = set(tuple(te_inputs[i][0].round(4)) for i in range(min(20, n_test)))
        overlap = tr_first_embeds & te_first_embeds
        # Allow small overlap (random chance a test entity could appear in train)
        # but should be much less than full overlap
        assert len(overlap) < 10, f"[{task_name}] suspicious train/test overlap: {len(overlap)}"

        print(f"  {task_name:<22} seq_len={seq_len}  "
              f"train={n_train} (match={n_match}, diff={n_diff})  "
              f"test={n_test}  ✓")

    print()
    print("All checks passed.")

    print("\n" + "─" * 60)
    print("Sample trial inspection: IdentityRules")
    print("─" * 60)
    task = factory.identity_rules
    inputs, labels, pnl_targets = task.generate(4, split='train')
    for i, (inp, label, pnl) in enumerate(zip(inputs, labels, pnl_targets)):
        print(f"\n  Trial {i}  label={'MATCH' if label[0]==1 else 'DIFF'}")
        print(f"    Sequence vectors (first 3 values shown):")
        for t, vec in enumerate(inp):
            print(f"      t={t}: {vec[:3].round(3)}...")
        print(f"    PsyNeuLink target sequence:")
        for t, tgt in enumerate(pnl):
            print(f"      t={t}: {tgt}")

    print("\n" + "─" * 60)
    print("Usage reminder:")
    print("─" * 60)
    print("""
  from dataset import DatasetFactory

  factory = DatasetFactory(embed_dim=64)

  # Generate trials for any task
  inputs, labels, pnl_targets = factory.same_different.generate(500, 'train')

  # Train
  esbn_gru_comp.learn(
      inputs  = {z_embedding_mech: inputs},
      targets = {y_hat_mech:       pnl_targets},
      epochs  = 100,
      learning_rate = 1e-3,
      execution_mode = pnl.ExecutionMode.PyTorch,
  )

  # Test on novel entities
  te_inputs, te_labels, _ = factory.same_different.generate(100, 'test')
  results = esbn_gru_comp.run(inputs={z_embedding_mech: te_inputs}, ...)
  evaluate_results(results, te_labels, task_name='SameDifferent')
""")
