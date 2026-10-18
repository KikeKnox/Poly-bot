"""rl_predictor v1 contract.

Names and order are EXACT literals: do not rename them. The feature vector
is float64, ordered under feature_schema='rl_predictor.features.v1'. The
YES/NO identity of a token is derived from market metadata
(yes_token_id/no_token_id), NEVER from its position in the actions or book
lists: the mask omits sides without a legal action, and positional
inference misclassifies them. Token identity comes from metadata, not from
the index or order of labels.
"""

FEATURE_SCHEMA = "rl_predictor.features.v1"
POLICY_ALGORITHM = "reinforce_linear_v1"
POLICY_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
ONLINE_MANIFEST_SCHEMA_VERSION = 1

# Contractual order of actions. The index IS the id.
ACTION_NAMES = (
    "HOLD",
    "BUY_YES_SMALL",
    "BUY_YES_FULL",
    "BUY_NO_SMALL",
    "BUY_NO_FULL",
    "SELL_YES_HALF",
    "SELL_YES_ALL",
    "SELL_NO_HALF",
    "SELL_NO_ALL",
)
NUM_ACTIONS = len(ACTION_NAMES)
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}

# Contractual order of features: 22 float64 values.
FEATURE_NAMES = (
    "p_yes",
    "p_missing",
    "yes_bid",
    "yes_ask",
    "no_bid",
    "no_ask",
    "yes_bid_depth_usdc",
    "yes_ask_depth_usdc",
    "no_bid_depth_usdc",
    "no_ask_depth_usdc",
    "time_remaining_frac",
    "yes_position_shares",
    "no_position_shares",
    "yes_cost_basis_usdc",
    "no_cost_basis_usdc",
    "free_budget_usdc",
    "reserved_budget_usdc",
    "pending_order",
    "yes_book_missing",
    "no_book_missing",
    "book_age_seconds",
    "buy_fee_rate",
)
NUM_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Accepted artifact statuses. No other value is allowed.
STATUS_CANDIDATE = "candidate"
STATUS_PAPER_APPROVED = "paper_approved"
# Online promotion is recorded as online_candidate and requires a valid
# online manifest (parent + hash + update chain). It is never recorded as
# paper_approved.
STATUS_ONLINE_CANDIDATE = "online_candidate"
VALID_STATUSES = (
    STATUS_CANDIDATE,
    STATUS_PAPER_APPROVED,
    STATUS_ONLINE_CANDIDATE,
)
# Statuses that authorize activating the strategy at runtime.
ACTIVATABLE_STATUSES = (STATUS_PAPER_APPROVED, STATUS_ONLINE_CANDIDATE)

# Mask: bit i = action i is legal. HOLD is always legal.
MASK_HOLD = 1 << ACTION_INDEX["HOLD"]

# Economic constants of the contract.
EPISODE_WINDOW_MS = 300_000  # horizon 5 min; time_remaining_frac = rem/300s
SMALL_FRACTION_OF_FULL = 0.5  # SMALL = half of the FULL budget


class ContractError(ValueError):
    """Contract v1 violation (fail-closed, never silently corrected)."""


def check_mask(mask):
    """Validates an action mask given as a bit field or a list of bools.

    Returns the bit field. Rejects: an illegal HOLD (it must always be
    legal), bits out of range, and unsupported types."""
    if isinstance(mask, bool):
        raise ContractError("scalar bool mask is not valid")
    if isinstance(mask, int):
        if mask < 0 or mask >= (1 << NUM_ACTIONS):
            raise ContractError(f"mask {mask} out of range for {NUM_ACTIONS} actions")
        bits = mask
    elif isinstance(mask, (list, tuple)):
        if len(mask) != NUM_ACTIONS:
            raise ContractError(
                f"mask has {len(mask)} entries, expected {NUM_ACTIONS}")
        bits = 0
        for i, v in enumerate(mask):
            if v not in (0, 1, False, True):
                raise ContractError(f"mask[{i}]={v!r} is not 0/1")
            if v:
                bits |= 1 << i
    else:
        raise ContractError(f"unsupported mask kind: {type(mask).__name__}")
    if not bits & MASK_HOLD:
        raise ContractError("mask without HOLD: HOLD is always legal")
    return bits


def legal_action_ids(mask):
    """IDs of the legal actions, in ascending contractual order."""
    bits = check_mask(mask)
    return [i for i in range(NUM_ACTIONS) if bits & (1 << i)]


def token_side(token_id, yes_token_id, no_token_id):
    """YES/NO identity of a token, from METADATA and never from position.

    Returns 'yes' | 'no'. Missing or conflicting metadata is not guessed:
    returns None, and the caller decides (block or mark as unknown)."""
    if token_id is None:
        return None
    tid = str(token_id)
    is_yes = yes_token_id is not None and tid == str(yes_token_id)
    is_no = no_token_id is not None and tid == str(no_token_id)
    if is_yes and is_no:
        return None  # contradictory metadata: do not pick a winner
    if is_yes:
        return "yes"
    if is_no:
        return "no"
    return None


def check_observation_vector(vec):
    """Validates an observation vector: 22 finite features.

    Returns a list of floats. NaN or inf invalidates the whole observation
    and any artifact built from it."""
    import math

    if not isinstance(vec, (list, tuple)) or len(vec) != NUM_FEATURES:
        raise ContractError(
            f"observation has {0 if vec is None else len(vec)} features, "
            f"expected {NUM_FEATURES} ({FEATURE_SCHEMA})")
    out = []
    for i, v in enumerate(vec):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ContractError(
                f"feature {FEATURE_NAMES[i]}={v!r} is not numeric")
        if not math.isfinite(f):
            raise ContractError(
                f"feature {FEATURE_NAMES[i]}={f!r} is not finite")
        out.append(f)
    return out
