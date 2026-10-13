"""Translation-invariant lateral contraction observations for inversion.

Positive observations denote shrinkage: -(u_high-u_low)/span. Geometry/pairs
are constructed outside autodiff; displacement evaluation stays in JAX.
"""
from dataclasses import dataclass
import numpy as onp
import jax.numpy as np


@dataclass
class RelativeContraction:
    pairs: tuple
    reference: object
    reduction: str
    penalty: str
    scale: float
    target_factor: float
    raw_reference: object = None

    def observe(self, displacement):
        return np.stack([
            -np.mean(displacement[high, axis] - displacement[low, axis]) / span
            for axis, low, high, span in self.pairs
        ])

    def residual(self, displacement):
        predicted = self.observe(displacement)
        target = self.reference * self.target_factor
        if self.reduction == "xy_mean":
            predicted, target = np.mean(predicted), np.mean(target)
        residual = (target - predicted) / self.scale
        if self.penalty == "under_only":
            residual = np.maximum(residual, 0.0)
        return residual

    def loss(self, displacement):
        return np.mean(self.residual(displacement) ** 2)

    def report(self, displacement):
        observed = onp.asarray(self.observe(displacement), dtype=float)
        reference = onp.asarray(self.reference, dtype=float)
        raw_reference = onp.asarray(
            self.reference if self.raw_reference is None else self.raw_reference,
            dtype=float,
        )
        return {
            "definition": "positive shrinkage: -mean(u_high-u_low)/span",
            "reduction": self.reduction, "penalty": self.penalty,
            "normalization_scale": self.scale, "target_factor": self.target_factor,
            "axes": {
                name: {"reference": float(reference[i]),
                       "raw_reference": float(raw_reference[i]),
                       "target": float(reference[i]*self.target_factor),
                       "simulated": float(observed[i]),
                       "paired_nodes": len(self.pairs[i][1]),
                       "span_m": float(self.pairs[i][3])}
                for i, name in enumerate(("x", "y"))
            },
            "raw_reference_xy_mean": float(raw_reference.mean()),
            "reference_expansion_clamped": bool(onp.any(reference != raw_reference)),
            "reference_xy_mean": float(reference.mean()),
            "target_xy_mean": float(reference.mean()*self.target_factor),
            "simulated_xy_mean": float(observed.mean()),
            "normalized_penalty": float(self.loss(displacement)),
        }


def build_relative_contraction(points, node_indices, reference_u, config):
    """Match opposite ROI faces by their transverse coordinates.

    Requires a matching face sampling rather than silently comparing different
    patches. Defaults to an xy-averaged observable for tied isotropic strain.
    For under_only, clamp expanding reference axes to zero before reduction;
    symmetric keeps the signed reference for two-sided deformation fitting.
    """
    if not config.get("enabled", False):
        return None
    reduction = config.get("reduction", "xy_mean")
    penalty = config.get("penalty", "under_only")
    if reduction not in ("xy_mean", "separate"):
        raise ValueError("relative_contraction.reduction must be xy_mean or separate")
    if penalty not in ("under_only", "symmetric"):
        raise ValueError("relative_contraction.penalty must be under_only or symmetric")
    tolerance = float(config.get("pair_tolerance_m", 1e-9))
    floor = float(config.get("scale_floor", 1e-5))
    target_factor = float(config.get("target_factor", 1.0))
    if not all(onp.isfinite(x) and x > 0 for x in (tolerance, floor, target_factor)):
        raise ValueError("Pair tolerance, scale floor and target factor must be finite and positive")
    points = onp.asarray(points)
    indices = onp.asarray(node_indices, dtype=int)
    if len(indices) == 0:
        raise ValueError("Relative contraction ROI is empty")
    roi_points = points[indices]
    pairs = []
    for axis in (0, 1):
        lo, hi = roi_points[:, axis].min(), roi_points[:, axis].max()
        span = float(hi-lo)
        if span <= 2*tolerance:
            raise ValueError("Relative contraction ROI has zero/ambiguous lateral span")
        low = indices[onp.abs(roi_points[:, axis]-lo) <= tolerance]
        high = indices[onp.abs(roi_points[:, axis]-hi) <= tolerance]
        transverse = [i for i in range(3) if i != axis]
        def sort_face(ids):
            coords = points[ids][:, transverse]
            return ids[onp.lexsort((coords[:, 1], coords[:, 0]))]
        low, high = sort_face(low), sort_face(high)
        if len(low) != len(high) or not onp.allclose(
            points[low][:,transverse], points[high][:,transverse], atol=tolerance, rtol=0
        ):
            raise ValueError("Opposite ROI faces do not have matching node pairs")
        pairs.append((axis, low, high, span))
    observation = RelativeContraction(tuple(pairs), None, reduction, penalty, 1., target_factor)
    reference = onp.asarray(observation.observe(np.asarray(reference_u)), dtype=float)
    if not onp.all(onp.isfinite(reference)):
        raise ValueError("Relative contraction reference is not finite")
    observation.raw_reference = np.asarray(reference)
    if penalty == "under_only":
        # A shrinking reference sets a lower-bound contraction target. An
        # expanding (negative) reference instead sets a zero target, without
        # changing AM displacements used by the other objective terms.
        reference = onp.maximum(reference, 0.0)
    scale = abs(reference.mean()) if reduction == "xy_mean" else onp.sqrt(onp.mean(reference**2))
    observation.reference = np.asarray(reference)
    observation.scale = float(max(scale, floor))
    return observation
