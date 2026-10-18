"""rl_predictor — Python LAB package.

Contract v1: 9 actions and 22 features in literal order, linear softmax
policy, ledger-based equity reward, episodic on-policy REINFORCE over frozen
cohorts.

Stdlib only. Delivered artifacts have status 'candidate' at most —
paper_approved/online_candidate are never fabricated here.
"""

from .contract import (  # noqa: F401
    ACTION_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    NUM_ACTIONS,
    NUM_FEATURES,
    POLICY_ALGORITHM,
)

__version__ = "0.1.0"
