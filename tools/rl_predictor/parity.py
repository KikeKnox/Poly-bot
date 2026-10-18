"""Exact REINFORCE math: pure functions shared by the trainer and the
Python/Rust parity fixtures (EXECUTION: "B replicates the Rust/Python
math exactly").

Frozen definition of the update (reference for the Rust learner):

1. Cohort: complete episodes collected with ONE frozen policy.
2. Reward-to-go gamma=1: G_t = sum_{k>=t} r_k (normalized reward).
3. Baseline b: mean of the returns of episodes from PREVIOUS batches
   (independent of the current action; first batch b=0).
4. Advantage A_t = G_t − b, the same for all the actions of the step.
5. For each step and legal action j: g_logit[j] = A_t·(δ(j=a_t) − π_j).
   Illegal actions do not participate (mask already applied in π).
6. Accumulation: dW[a][f] += g_logit[a]·x_std[f]; db[a] += g_logit[a].
7. Mean over the total steps of the cohort.
8. Global L2 norm clip to grad_clip (uniform scale of the whole
   gradient; never clip per element).
9. param += lr · grad_clipped (ascent: maximizes the expected return).

Stdlib/float only: the model is float; money has already been normalized.
"""

import math

from .contract import NUM_ACTIONS, NUM_FEATURES
from .policy import LinearPolicy


def returns_to_go(rewards, gamma=1.0):
    """G_t = r_t + gamma·G_{t+1}, a single backward pass."""
    g = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        g[t] = acc
    return g


def cohort_gradients(cohort, policy, baseline, gamma=1.0):
    """Gradient of the update over a frozen cohort.

    cohort: episode items; each episode is a list of steps
    {x_std, mask, action, reward}. Returns
    (dW, db, n_steps, episode_returns). Any non-finite value in
    rewards/x_std aborts with ValueError (numeric update abort)."""
    dW = [[0.0] * NUM_FEATURES for _ in range(NUM_ACTIONS)]
    db = [0.0] * NUM_ACTIONS
    n_steps = 0
    ep_returns = []
    for ep in cohort:
        rewards = [float(s["reward"]) for s in ep]
        for r in rewards:
            if not math.isfinite(r):
                raise ValueError(f"non-finite reward {r}: update aborted")
        ep_returns.append(sum(rewards))
        gts = returns_to_go(rewards, gamma)
        for step, g_t in zip(ep, gts):
            # sampled=False (warmup/forced HOLD, argmax eval): the reward
            # already entered G_t of earlier steps, but this step contributes
            # no gradient and is not counted in n_steps — parity with learner.rs.
            if not step.get("sampled", True):
                continue
            x = [float(v) for v in step["x_std"]]
            if len(x) != NUM_FEATURES or any(
                    not math.isfinite(v) for v in x):
                raise ValueError("non-finite x_std: update aborted")
            adv = g_t - baseline
            probs = policy.probs(x, step["mask"])
            a_t = step["action"]
            if probs[a_t] <= 0.0:
                raise ValueError(
                    f"action {a_t} with π=0 in an on-policy cohort: "
                    "corrupt data or off-policy without correction")
            for j in range(NUM_ACTIONS):
                if probs[j] <= 0.0:
                    continue  # illegal: does not participate
                gl = adv * ((1.0 if j == a_t else 0.0) - probs[j])
                row = dW[j]
                for f_ in range(NUM_FEATURES):
                    row[f_] += gl * x[f_]
                db[j] += gl
            n_steps += 1
    if n_steps == 0:
        raise ValueError("cohort without steps: nothing to learn")
    inv = 1.0 / n_steps
    for a in range(NUM_ACTIONS):
        for f_ in range(NUM_FEATURES):
            dW[a][f_] *= inv
        db[a] *= inv
    return dW, db, n_steps, ep_returns


def clip_gradients(dW, db, max_norm):
    """Global L2 norm clip. Returns (dW, db, norm, scale_applied)."""
    if max_norm <= 0:
        raise ValueError("grad_clip must be > 0")
    sq = sum(v * v for row in dW for v in row) + sum(v * v for v in db)
    norm = math.sqrt(sq)
    if not math.isfinite(norm):
        raise ValueError("gradient norm not finite: update aborted")
    scale = 1.0 if norm <= max_norm else max_norm / norm
    if scale != 1.0:
        for a in range(NUM_ACTIONS):
            for f_ in range(NUM_FEATURES):
                dW[a][f_] *= scale
            db[a] *= scale
    return dW, db, norm, scale


def apply_gradients(policy, dW, db, lr):
    """param += lr·grad (delegated to LinearPolicy.apply_update, which
    validates finiteness after the update)."""
    policy.apply_update(dW, db, lr)


def full_update(cohort, policy, baseline, *, lr, grad_clip, gamma=1.0):
    """Full update over a copy: returns (new_policy, report)."""
    dW, db, n_steps, ep_returns = cohort_gradients(cohort, policy, baseline,
                                                 gamma)
    dW, db, norm, scale = clip_gradients(dW, db, grad_clip)
    new_policy = policy.copy()
    apply_gradients(new_policy, dW, db, lr)
    return new_policy, {
        "baseline": baseline,
        "n_steps": n_steps,
        "episode_returns": ep_returns,
        "grad_norm_preclip": norm,
        "clip_scale": scale,
        "lr": lr,
        "grad_clip": grad_clip,
        "weights": new_policy.weights,
        "bias": new_policy.bias,
    }
