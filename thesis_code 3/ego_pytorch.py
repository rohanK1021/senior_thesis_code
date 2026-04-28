"""
Pure PyTorch implementation of the EGO model from Giallanza et al. (2024) Study 2.

Recreates the PsyNeuLink model at:
    /Users/a/Downloads/Giallanza2024_ego_study2.py

Key components:
    - Episodic Memory (EM): content-addressable memory storing (state, prev_state, context) triplets
    - Context layer: integrates states over time via exponential moving average + tanh
    - Learnable context projection + bias: transforms context before querying EM
    - Prediction = retrieved state from EM
    - Loss: binary cross-entropy between prediction and next state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

STATE_SIZE = 10
CONTEXT_INTEGRATION_RATE = 0.7
PREVIOUS_STATE_RETRIEVAL_WEIGHT = 1.0
CONTEXT_RETRIEVAL_WEIGHT = 1.0
MEMORY_FILL = 0.001
SOFTMAX_TEMPERATURE = 0.2
SOFTMAX_THRESHOLD = 0.01
LEARNING_RATE = 2.0
NUM_OPTIM_STEPS = 5
LOSS_SPEC = "binary_cross_entropy"


# ── Episodic Memory ─────────────────────────────────────────────────────────

class EpisodicMemory:
    """Content-addressable memory that stores (state, prev_state, context) triplets.

    Retrieval is based on softmax-weighted similarity of query fields
    (prev_state and context) against stored entries. The state field is
    only stored/retrieved, never used as a query key.
    """

    def __init__(self, capacity, state_size, fill_value=0.001,
                 softmax_gain=5.0, softmax_threshold=0.01,
                 prev_state_weight=1.0, context_weight=1.0):
        self.capacity = capacity
        self.state_size = state_size
        self.softmax_gain = softmax_gain
        self.softmax_threshold = softmax_threshold
        self.prev_state_weight = prev_state_weight
        self.context_weight = context_weight

        # Memory banks: [capacity, state_size] — initialized with small fill value
        self.mem_state = torch.full((capacity, state_size), fill_value, dtype=torch.float64)
        self.mem_prev_state = torch.full((capacity, state_size), fill_value, dtype=torch.float64)
        self.mem_context = torch.full((capacity, state_size), fill_value, dtype=torch.float64)

        self.write_head = 0  # next slot to write to

    def retrieve(self, query_prev_state, query_context):
        """Retrieve state from memory using prev_state and context as queries.

        Args:
            query_prev_state: [state_size] tensor
            query_context: [state_size] tensor (already projected & normalized)

        Returns:
            retrieved_state: [state_size] tensor (differentiable w.r.t. query_context)
        """
        # Dot-product similarities: [capacity]
        sim_prev = torch.mv(self.mem_prev_state, query_prev_state)  # not differentiable (no learnable params)
        sim_ctx = torch.mv(self.mem_context, query_context)          # differentiable through query_context

        # Weighted combination of similarities
        combined_sim = self.prev_state_weight * sim_prev + self.context_weight * sim_ctx

        # Softmax with gain (= 1/temperature)
        weights = F.softmax(self.softmax_gain * combined_sim, dim=0)

        # Threshold: zero out small weights
        weights = weights * (weights > self.softmax_threshold).float()
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum  # re-normalize

        # Weighted retrieval of state
        retrieved = torch.mv(self.mem_state.t(), weights)  # [state_size]
        return retrieved

    def store(self, state, prev_state, context):
        """Store a triplet in memory (non-differentiable, uses detached values)."""
        idx = self.write_head % self.capacity
        self.mem_state[idx] = state.detach()
        self.mem_prev_state[idx] = prev_state.detach()
        self.mem_context[idx] = context.detach()
        self.write_head += 1

    def reset(self, fill_value=0.001):
        self.mem_state.fill_(fill_value)
        self.mem_prev_state.fill_(fill_value)
        self.mem_context.fill_(fill_value)
        self.write_head = 0


# ── EGO Model ───────────────────────────────────────────────────────────────

class EGOModel(nn.Module):
    """The EGO (Episodic Generalization and Optimization) model.

    Learnable parameters:
        - context_projection: [state_size, state_size] matrix mapping context → query
        - context_bias: [1, state_size] bias added after projection
    """

    def __init__(self, state_size=STATE_SIZE):
        super().__init__()
        self.state_size = state_size

        # Learnable parameters
        self.context_projection = nn.Parameter(
            torch.eye(state_size, dtype=torch.float64)
        )
        self.context_bias = nn.Parameter(
            torch.zeros(1, state_size, dtype=torch.float64)
        )

    def forward_context_query(self, context):
        """Transform context into a normalized query for EM.

        context -> matmul with learned projection -> add bias -> normalize
        """
        projected = context @ self.context_projection + self.context_bias.squeeze(0)
        # Normalize (L2)
        norm = torch.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected


# ── Training Loop ────────────────────────────────────────────────────────────

def run_ego(
    states,
    memory_capacity=None,
    state_size=STATE_SIZE,
    integration_rate=CONTEXT_INTEGRATION_RATE,
    prev_state_weight=PREVIOUS_STATE_RETRIEVAL_WEIGHT,
    context_weight=CONTEXT_RETRIEVAL_WEIGHT,
    memory_fill=MEMORY_FILL,
    softmax_temperature=SOFTMAX_TEMPERATURE,
    softmax_threshold=SOFTMAX_THRESHOLD,
    learning_rate=LEARNING_RATE,
    num_optim_steps=NUM_OPTIM_STEPS,
):
    """Run the EGO model on a sequence of one-hot states.

    The model processes states one at a time. At each trial:
        1. Use previous_state and context to retrieve a prediction from EM
        2. Compare prediction with current state (= target), compute loss
        3. Optimize the context projection (multiple gradient steps)
        4. Store (current_state, previous_state, context_query) in EM
        5. Update previous_state and context for next trial

    Args:
        states: list of one-hot encoded states, each of length state_size

    Returns:
        predictions: numpy array of shape [num_trials, state_size]
    """
    if memory_capacity is None:
        memory_capacity = len(states)

    softmax_gain = 1.0 / softmax_temperature

    # Initialize model and memory
    model = EGOModel(state_size)
    model.double()

    em = EpisodicMemory(
        capacity=memory_capacity,
        state_size=state_size,
        fill_value=memory_fill,
        softmax_gain=softmax_gain,
        softmax_threshold=softmax_threshold,
        prev_state_weight=prev_state_weight,
        context_weight=context_weight,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # State variables (not learnable, updated each trial)
    previous_state = torch.zeros(state_size, dtype=torch.float64)
    context = torch.zeros(state_size, dtype=torch.float64)

    all_predictions = []

    for t in range(len(states)):
        current_state = torch.tensor(states[t], dtype=torch.float64)

        # ── Step 1: Retrieve from EM (before updating context/prev_state) ──
        # Transform context into query
        context_query = model.forward_context_query(context)

        # Retrieve prediction from EM
        prediction = em.retrieve(previous_state, context_query)
        all_predictions.append(prediction.detach().numpy().copy())

        # ── Step 2: Compute loss and optimize (multiple steps) ──
        # Target is the current state
        target = current_state

        for step in range(num_optim_steps):
            optimizer.zero_grad()

            # Re-compute prediction (for gradient flow)
            ctx_q = model.forward_context_query(context)
            pred = em.retrieve(previous_state, ctx_q)

            # Binary cross-entropy loss
            # Clamp prediction to avoid log(0)
            pred_clamped = torch.clamp(pred, 1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(pred_clamped, target)

            loss.backward()
            optimizer.step()

        # ── Step 3: Store in EM (after optimization, using final context query) ──
        with torch.no_grad():
            final_ctx_q = model.forward_context_query(context)
        # Store on last optimization step (matches PNL's store_on_optimization=LAST)
        em.store(current_state, previous_state, final_ctx_q)

        # ── Step 4: Update state variables for next trial ──
        previous_state = current_state.detach().clone()

        # Context integration: context = tanh((1-rate)*context + rate*state)
        context = torch.tanh(
            (1 - integration_rate) * context.detach() + integration_rate * current_state.detach()
        )

    return np.array(all_predictions)


# ── Data Generation (same as PNL script) ────────────────────────────────────

def to_one_hot(state, num_states=10):
    one_hot = np.eye(num_states)
    return list(one_hot[state - 1])


def get_trials(paradigm, n_contexts, n_blocks=None):
    def gen_context_1():
        states = [9, np.random.choice([1, 2])]
        for _ in range(3):
            states.append(states[-1] + 2)
        return states

    def gen_context_2():
        states = [10, np.random.choice([1, 2])]
        for _ in range(3):
            if states[-1] % 2 == 0:
                states.append(states[-1] + 1)
            else:
                states.append(states[-1] + 3)
        return states

    contexts = []
    if paradigm == 'interleaved':
        for i in range(n_contexts):
            if i % 2 == 0:
                contexts += gen_context_1()
            else:
                contexts += gen_context_2()
    elif paradigm == 'blocked':
        assert n_blocks is not None and n_blocks % 2 == 0
        assert n_contexts % n_blocks == 0
        for i in range(n_blocks):
            for _ in range(n_contexts // n_blocks):
                if i % 2 == 0:
                    contexts += gen_context_1()
                else:
                    contexts += gen_context_2()

    # Test phase: n_contexts // 4 random sequences
    for _ in range(n_contexts // 4):
        if np.random.random() < 0.5:
            contexts += gen_context_1()
        else:
            contexts += gen_context_2()

    return [to_one_hot(c) for c in contexts]


# ── Analysis ─────────────────────────────────────────────────────────────────

def calc_prob(em_preds, test_ys):
    """Probability of correct next-state prediction (terminal 3 states per sequence)."""
    em_preds_new = em_preds[:, 2:-1, :]
    test_ys_new = test_ys[:, 2:-1, :]
    em_probability = (em_preds_new * test_ys_new).sum(-1).mean(-1)
    return em_probability


def get_df(results, states, participant_nr, paradigm, state_size=STATE_SIZE):
    em_preds = results.reshape(-1, 5, state_size)
    test_ys = np.array(states).reshape(-1, 5, state_size)
    correct_prob = calc_prob(em_preds, test_ys)

    return pd.DataFrame({
        'participant_nr': [participant_nr] * len(correct_prob),
        'paradigm': [paradigm] * len(correct_prob),
        'trial': list(range(len(correct_prob))),
        'probability': correct_prob,
    })


def run_experiment(n_participants, n_contexts, n_blocks):
    all_data = []
    for p in range(n_participants):
        print(f'Running participant {p}')
        for paradigm in ['interleaved', 'blocked']:
            states = get_trials(paradigm=paradigm, n_contexts=n_contexts, n_blocks=n_blocks)
            results = run_ego(states, memory_capacity=len(states))
            df = get_df(results, states, participant_nr=p, paradigm=paradigm)
            all_data.append(df)
    return pd.concat(all_data).reset_index(drop=True)


def plot_results(df, plot_title, n_contexts, n_blocks):
    palette = {"interleaved": "grey", "blocked": "black"}
    fig, ax = plt.subplots(figsize=(8, 4))

    for paradigm, color in palette.items():
        d = df[df["paradigm"] == paradigm]
        mean = d.groupby("trial")["probability"].mean()
        se = d.groupby("trial")["probability"].sem()
        ax.plot(mean.index, mean.values, color=color, label=paradigm)
        ax.fill_between(mean.index, mean - se, mean + se, color=color, alpha=0.2)

    if n_blocks is not None:
        block_size = n_contexts // n_blocks
        for block in range(n_blocks):
            start = block * block_size
            end = (block + 1) * block_size
            color = "blue" if block % 2 == 0 else "orange"
            ax.axvspan(start, end, color=color, alpha=0.08)

    test_start = n_contexts
    test_end = n_contexts + (n_contexts // 4)
    ax.axvspan(test_start, test_end, color="green", alpha=0.10)

    ax.set_xlim(0, n_contexts + (n_contexts // 4))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(correct)")
    ax.set_title(plot_title)
    ax.legend()
    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_CONTEXTS = 200
    N_BLOCKS = 4
    N_PARTICIPANTS = 10

    data = run_experiment(n_participants=N_PARTICIPANTS, n_contexts=N_CONTEXTS, n_blocks=N_BLOCKS)
    fig = plot_results(data, 'EGO Model (PyTorch)', n_contexts=N_CONTEXTS, n_blocks=N_BLOCKS)
    plt.savefig('ego_pytorch_results.png', dpi=150)
    plt.show()
