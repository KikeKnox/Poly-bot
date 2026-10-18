"""Linear softmax policy with INDEPENDENT weights per action
logits[a] = bias[a] + dot(weights[a], x_std).

Mask applied BEFORE softmax; the max of the legal logits is subtracted for
numerical stability. Adaptive paper training: sampling with a seed and the
recorded behaviour probability. Runtime/evaluation is frozen:
deterministic legal argmax, ties broken by index (HOLD first).

Stdlib only: no numpy. Money/reward lives in reward.py with Decimal; the
model works in float, as the contract requires.
"""

import math
import random

from .contract import (
    NUM_ACTIONS,
    NUM_FEATURES,
    ContractError,
    check_mask,
    check_observation_vector,
)


class PolicyError(ContractError):
    pass


def _check_params(weights, bias):
    if len(weights) != NUM_ACTIONS or len(bias) != NUM_ACTIONS:
        raise PolicyError(
            f"weights {len(weights)}x?/bias {len(bias)}, expected "
            f"{NUM_ACTIONS}x{NUM_FEATURES}/{NUM_ACTIONS}")
    for a in range(NUM_ACTIONS):
        if len(weights[a]) != NUM_FEATURES:
            raise PolicyError(
                f"weights[{a}] has {len(weights[a])} features, "
                f"expected {NUM_FEATURES}")
        for f, w in enumerate(weights[a]):
            if not math.isfinite(w):
                raise PolicyError(f"weights[{a}][{f}]={w} not finite")
        if not math.isfinite(bias[a]):
            raise PolicyError(f"bias[{a}]={bias[a]} not finite")


class LinearPolicy:
    """Masked linear softmax. Immutable except through `apply_update`
    of the trainer (gradient already clipped and finite)."""

    def __init__(self, weights=None, bias=None):
        if weights is None:
            weights = [[0.0] * NUM_FEATURES for _ in range(NUM_ACTIONS)]
        if bias is None:
            bias = [0.0] * NUM_ACTIONS
        _check_params(weights, bias)
        self.weights = [list(map(float, row)) for row in weights]
        self.bias = list(map(float, bias))

    def copy(self):
        return LinearPolicy(self.weights, self.bias)

    def logits(self, x_std):
        """Raw logits per action (no mask)."""
        x = check_observation_vector(x_std)
        out = []
        for a in range(NUM_ACTIONS):
            w = self.weights[a]
            out.append(self.bias[a] + sum(wf * xf for wf, xf in zip(w, x)))
        return out

    def probs(self, x_std, mask):
        """Softmax over LEGAL actions. Returns NUM_ACTIONS items
        with 0.0 for illegal actions. Stable: subtracts the legal max.

        Illegal actions get probability exactly 0 —
        no epsilon residue, no redistribution."""
        bits = check_mask(mask)
        lg = self.logits(x_std)
        legal = [a for a in range(NUM_ACTIONS) if bits & (1 << a)]
        m = max(lg[a] for a in legal)
        exps = {a: math.exp(lg[a] - m) for a in legal}
        z = sum(exps.values())
        probs = [0.0] * NUM_ACTIONS
        for a, e in exps.items():
            probs[a] = e / z
        return probs

    def argmax(self, x_std, mask):
        """Runtime action: deterministic legal argmax; ties -> the lowest index
        (HOLD=0 wins)."""
        probs = self.probs(x_std, mask)
        best, best_p = 0, -1.0
        for a in range(NUM_ACTIONS):
            if probs[a] > best_p:
                best, best_p = a, probs[a]
        return best

    def sample(self, x_std, mask, rng):
        """Sampling for adaptive PAPER training.

        rng: random.Random seeded by the caller (reproducibility per
        episode/batch, recorded outside). Returns (action_id, prob)
        with prob = behaviour probability to record."""
        if not isinstance(rng, random.Random):
            raise PolicyError("sample requires a seeded random.Random")
        probs = self.probs(x_std, mask)
        u = rng.random()
        acc = 0.0
        last_legal = None
        for a in range(NUM_ACTIONS):
            if probs[a] > 0.0:
                last_legal = a
                acc += probs[a]
                if u < acc:
                    return a, probs[a]
        # Float sum slightly below 1 due to rounding: falls into the last legal action.
        return last_legal, probs[last_legal]

    def log_prob(self, action, x_std, mask):
        """log π(a|x,mask). Rejects an illegal action (non-finite: error, not
        a silent -inf that contaminates gradients)."""
        probs = self.probs(x_std, mask)
        p = probs[action]
        if p <= 0.0:
            raise PolicyError(
                f"action {action} illegal under mask: log_prob undefined")
        return math.log(p)

    def grad_log_probs(self, x_std, mask):
        """∇_logits log π(a|·) for every legal action: δ(j=a) − π_j.

        Returns a [NUM_ACTIONS][NUM_ACTIONS] matrix: row a = grad of
        log π(a) w.r.t. logits[0..8]. Rows of illegal actions are
        None (they do not participate). Useful for exact REINFORCE and parity checks."""
        probs = self.probs(x_std, mask)
        grads = [None] * NUM_ACTIONS
        for a in range(NUM_ACTIONS):
            if probs[a] > 0.0:
                grads[a] = [
                    (1.0 if j == a else 0.0) - probs[j]
                    for j in range(NUM_ACTIONS)
                ]
        return grads

    def apply_update(self, d_weights, d_bias, lr):
        """Applies an already clipped gradient step:
        param += lr * grad (REINFORCE maximizes return: the caller passes the
        policy gradient with the maximization sign)."""
        if len(d_weights) != NUM_ACTIONS or len(d_bias) != NUM_ACTIONS:
            raise PolicyError("invalid update dimensions")
        for a in range(NUM_ACTIONS):
            for f in range(NUM_FEATURES):
                g = d_weights[a][f]
                if not math.isfinite(g):
                    raise PolicyError("non-finite gradient: update aborted")
                self.weights[a][f] += lr * g
            gb = d_bias[a]
            if not math.isfinite(gb):
                raise PolicyError("non-finite bias gradient: update aborted")
            self.bias[a] += lr * gb
        # Post-update check: the artifact never ends up non-finite.
        _check_params(self.weights, self.bias)
