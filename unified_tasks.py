"""
Unified Task Suite — Fixed Sequence Length
===========================================
All 4 tasks use seq_len=6 and binary output (output_dim=2).
Context (task_id) is the ONLY way to know which rule applies.
The same entity sequence can require different answers under different rules.

Tasks:
  0: Same/Different — Are positions 0,1 identical? (2-5 distractors)
  1: RMTS — Does rel(0,1)==rel(2,3)? (4-5 distractors)
  2: Identity Rules — Do triples (0,1,2) and (3,4,5) have same ABA/ABB pattern?
  3: Distribution-of-3 — Do (0,1,2) share an attribute? Does (3) share it? (4-5 distractors)

Uses the existing entity system from dataset.py.
"""

import numpy as np
from dataset import Entity, EntityPool, ALL_ENTITIES, ATTR_SIZES, ATTR_NAMES

SEQ_LEN = 6
OUTPUT_DIM = 2
N_TASKS = 4
TASK_NAMES = ['same_different', 'rmts', 'identity_rules', 'dist3']
TASK_SHORTS = ['S/D', 'RMTS', 'IdRules', 'Dist3']


class UnifiedTaskFactory:
    """Generates training/test batches for all 4 tasks at fixed seq_len=6."""

    def __init__(self, train_entities, test_entities, seed=42):
        self.train_entities = train_entities
        self.test_entities = test_entities
        self.rng = np.random.RandomState(seed)
        self.task_names = TASK_NAMES
        self.num_tasks = N_TASKS

    def _random_entities(self, entities, n):
        """Sample n random entities (with replacement)."""
        idxs = self.rng.randint(len(entities), size=n)
        return [entities[i] for i in idxs]

    def _different_entity(self, entities, exclude):
        """Sample an entity different from exclude."""
        pool = [e for e in entities if e != exclude]
        return pool[self.rng.randint(len(pool))]

    def _entities_with_attr(self, entities, attr_idx, val):
        return [e for e in entities if e[attr_idx] == val]

    def _entities_without_attr(self, entities, attr_idx, val):
        return [e for e in entities if e[attr_idx] != val]

    # ── Task 0: Same/Different ──
    def _gen_same_different(self, entities, is_match):
        """Positions 0,1 = pair. Positions 2-5 = random distractors."""
        a = entities[self.rng.randint(len(entities))]
        if is_match:
            b = a
        else:
            b = self._different_entity(entities, a)
        distractors = self._random_entities(entities, 4)
        seq = [a, b] + distractors
        return seq

    # ── Task 1: RMTS ──
    def _gen_rmts(self, entities, is_match):
        """Positions 0,1 = pair1. Positions 2,3 = pair2. 4,5 = distractors.
        Match = rel(pair1)==rel(pair2). Both same or both different."""
        pair1_same = self.rng.random() < 0.5
        if pair1_same:
            a = entities[self.rng.randint(len(entities))]
            b = a
        else:
            idxs = self.rng.choice(len(entities), 2, replace=False)
            a, b = entities[idxs[0]], entities[idxs[1]]

        if is_match:
            pair2_same = pair1_same
        else:
            pair2_same = not pair1_same

        if pair2_same:
            c = entities[self.rng.randint(len(entities))]
            d = c
        else:
            idxs = self.rng.choice(len(entities), 2, replace=False)
            c, d = entities[idxs[0]], entities[idxs[1]]

        distractors = self._random_entities(entities, 2)
        seq = [a, b, c, d] + distractors
        return seq

    # ── Task 2: Identity Rules ──
    def _gen_identity_rules(self, entities, is_match):
        """Positions 0,1,2 = triple1. Positions 3,4,5 = triple2.
        Each triple is ABA or ABB. Match = same pattern."""
        # Triple 1
        pattern1 = 'ABA' if self.rng.random() < 0.5 else 'ABB'
        idxs = self.rng.choice(len(entities), 2, replace=False)
        p1, p2 = entities[idxs[0]], entities[idxs[1]]
        if pattern1 == 'ABA':
            triple1 = [p1, p2, p1]
        else:
            triple1 = [p1, p2, p2]

        # Triple 2
        if is_match:
            pattern2 = pattern1
        else:
            pattern2 = 'ABB' if pattern1 == 'ABA' else 'ABA'

        idxs = self.rng.choice(len(entities), 2, replace=False)
        q1, q2 = entities[idxs[0]], entities[idxs[1]]
        if pattern2 == 'ABA':
            triple2 = [q1, q2, q1]
        else:
            triple2 = [q1, q2, q2]

        seq = triple1 + triple2
        return seq

    # ── Task 3: Distribution-of-3 ──
    def _gen_dist3(self, entities, is_match):
        """Positions 0,1,2 = three entities sharing an attribute.
        Position 3 = test (shares or doesn't). Positions 4,5 = distractors."""
        # Pick rule attribute and value
        attr_idx = self.rng.randint(len(ATTR_SIZES))
        n_vals = ATTR_SIZES[attr_idx]
        rule_val = self.rng.randint(n_vals)

        pool_in = self._entities_with_attr(entities, attr_idx, rule_val)
        pool_out = self._entities_without_attr(entities, attr_idx, rule_val)

        # Need at least 3 entities with the attribute
        if len(pool_in) < 3:
            # Fallback: try another attribute
            for alt_attr in range(len(ATTR_SIZES)):
                for alt_val in range(ATTR_SIZES[alt_attr]):
                    alt_pool = self._entities_with_attr(entities, alt_attr, alt_val)
                    if len(alt_pool) >= 3:
                        pool_in = alt_pool
                        pool_out = self._entities_without_attr(entities, alt_attr, alt_val)
                        break
                if len(pool_in) >= 3:
                    break

        idxs = self.rng.choice(len(pool_in), 3, replace=False)
        a, b, c = pool_in[idxs[0]], pool_in[idxs[1]], pool_in[idxs[2]]

        if is_match:
            remaining = [e for e in pool_in if e not in {a, b, c}]
            if remaining:
                d = remaining[self.rng.randint(len(remaining))]
            else:
                d = pool_in[self.rng.randint(len(pool_in))]
        else:
            d = pool_out[self.rng.randint(len(pool_out))]

        distractors = self._random_entities(entities, 2)
        seq = [a, b, c, d] + distractors
        return seq

    # ── Unified interface ──
    def generate_batch(self, task_id, batch_size, split='train'):
        """Generate a batch of trials for the given task.

        Returns:
            inputs: ndarray (batch_size, 6, 12) — one-hot encoded entities
            targets: ndarray (batch_size, 2) — binary labels
        """
        entities = self.train_entities if split == 'train' else self.test_entities
        generators = [
            self._gen_same_different,
            self._gen_rmts,
            self._gen_identity_rules,
            self._gen_dist3,
        ]
        gen = generators[task_id]

        inputs = []
        targets = []
        n_match = batch_size // 2
        n_nonmatch = batch_size - n_match

        for i in range(batch_size):
            is_match = (i < n_match)
            seq = gen(entities, is_match)
            one_hots = np.stack([e.one_hot() for e in seq])
            inputs.append(one_hots)
            if is_match:
                targets.append(np.array([1., 0.], dtype=np.float32))
            else:
                targets.append(np.array([0., 1.], dtype=np.float32))

        inputs = np.array(inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        # Shuffle
        perm = self.rng.permutation(batch_size)
        return inputs[perm], targets[perm]

    @staticmethod
    def compute_answer(task_id, entities):
        """Compute the correct binary answer (0=match, 1=nonmatch) for a
        sequence of 6 entities under the given rule.

        Returns None if the answer is undefined (e.g., Dist3 with no shared attribute).
        """
        if task_id == 0:  # Same/Different: are positions 0,1 identical?
            return 0 if entities[0] == entities[1] else 1

        elif task_id == 1:  # RMTS: rel(0,1) == rel(2,3)?
            rel1 = (entities[0] == entities[1])
            rel2 = (entities[2] == entities[3])
            return 0 if rel1 == rel2 else 1

        elif task_id == 2:  # Identity Rules: pattern(0,1,2) == pattern(3,4,5)?
            def get_pattern(a, b, c):
                if a == c and a != b:
                    return 'ABA'
                elif b == c and a != b:
                    return 'ABB'
                elif a == b == c:
                    return 'AAA'
                else:
                    return 'OTHER'
            p1 = get_pattern(entities[0], entities[1], entities[2])
            p2 = get_pattern(entities[3], entities[4], entities[5])
            if p1 == 'OTHER' or p2 == 'OTHER':
                return None
            return 0 if p1 == p2 else 1

        elif task_id == 3:  # Dist3: do (0,1,2) share an attribute? Does (3)?
            for attr_idx in range(len(ATTR_SIZES)):
                val0 = entities[0][attr_idx]
                if entities[1][attr_idx] == val0 and entities[2][attr_idx] == val0:
                    # Found shared attribute
                    return 0 if entities[3][attr_idx] == val0 else 1
            return None  # No shared attribute found

        return None
