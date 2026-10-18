"""Policy scaler: train-only, mean/scale per feature,
scale>0, standardized clamp [-10,10]. NaN/inf invalidate the observation.

`fit` ONLY over train observations; `transform` never mutates the
parameters. The frozen predictor's scaler is NOT reused or touched:
this vector is exclusive to the policy.
"""

import math

from .contract import (
    FEATURE_NAMES,
    NUM_FEATURES,
    ContractError,
    check_observation_vector,
)

CLAMP_STD = 10.0


class ScalerError(ContractError):
    pass


class FeatureScaler:
    """z-score per feature with frozen parameters after fit.

    scale>0 always: if the train variance is 0 the scale stays at 1.0
    (constant feature -> 0 after centering), never divides by zero."""

    def __init__(self, mean, scale):
        if len(mean) != NUM_FEATURES or len(scale) != NUM_FEATURES:
            raise ScalerError(
                f"scaler dims {len(mean)}/{len(scale)}, expected "
                f"{NUM_FEATURES}/{NUM_FEATURES}")
        self.mean = [float(m) for m in mean]
        self.scale = [float(s) for s in scale]
        for i, (m, s) in enumerate(zip(self.mean, self.scale)):
            if not math.isfinite(m) or not math.isfinite(s):
                raise ScalerError(
                    f"scaler[{FEATURE_NAMES[i]}] mean/scale not finite")
            if s <= 0.0:
                raise ScalerError(
                    f"scaler[{FEATURE_NAMES[i]}] scale={s} <= 0")

    @classmethod
    def fit(cls, observations):
        """Fits mean/scale over TRAIN observations exclusively.

        observations: iterable of 22-feature vectors (already validated or
        raw — they are validated here; a non-finite observation aborts the fit,
        it is not silently imputed."""
        rows = [check_observation_vector(o) for o in observations]
        if not rows:
            raise ScalerError("scaler fit without train observations")
        n = len(rows)
        mean = [sum(r[i] for r in rows) / n for i in range(NUM_FEATURES)]
        scale = []
        for i in range(NUM_FEATURES):
            var = sum((r[i] - mean[i]) ** 2 for r in rows) / n
            s = math.sqrt(var)
            scale.append(s if s > 0.0 else 1.0)
        return cls(mean, scale)

    def transform(self, vec):
        """Standardizes and clamps to [-10,10]. Does not mutate self.

        Rejects NaN/inf input (they invalidate the observation)."""
        v = check_observation_vector(vec)
        out = []
        for x, m, s in zip(v, self.mean, self.scale):
            z = (x - m) / s
            if z > CLAMP_STD:
                z = CLAMP_STD
            elif z < -CLAMP_STD:
                z = -CLAMP_STD
            out.append(z)
        return out

    def to_dict(self):
        return {"mean": list(self.mean), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or "mean" not in d or "scale" not in d:
            raise ScalerError("scaler artifact without mean/scale")
        return cls(d["mean"], d["scale"])
