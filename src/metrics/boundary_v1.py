"""Inner-boundary F1 with Euclidean pixel tolerance and masked edges."""
import numpy as np
from scipy import ndimage


def boundary_counts(prediction, truth, valid, tolerance=2):
    p, y, v = map(np.asarray, (prediction, truth, valid))
    if p.ndim != 2 or p.shape != y.shape or p.shape != v.shape:
        raise ValueError("Matching two-dimensional arrays required")
    if v.dtype != np.bool_:
        raise ValueError("Boolean valid mask required")
    if not np.isin(p, (0, 1)).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Binary prediction and truth required")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("Nonnegative integer tolerance required")

    p, y = p.astype(bool), y.astype(bool)
    structure = np.ones((3, 3), dtype=bool)

    # One pixel for the boundary operator; additional tolerance-wide exclusion
    # prevents unknown pixels and coverage edges from producing scored matches.
    safe = ndimage.binary_erosion(
        v, structure=structure, iterations=1, border_value=0
    )
    scored = ndimage.binary_erosion(
        v, structure=structure, iterations=tolerance+1, border_value=0
    )

    def inner_boundary(mask):
        return (
            mask
            & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)
            & safe
        )

    bp, by = inner_boundary(p), inner_boundary(y)
    scored_p, scored_y = bp & scored, by & scored

    def matched(source, target):
        if not target.any():
            return 0
        distance = ndimage.distance_transform_edt(~target)
        return int((source & (distance <= tolerance)).sum())

    return {
        "predicted_boundary_pixels": int(scored_p.sum()),
        "true_boundary_pixels": int(scored_y.sum()),
        "matched_predicted_pixels": matched(scored_p, by),
        "matched_true_pixels": matched(scored_y, bp),
        "scored_support_pixels": int(scored.sum()),
    }


def boundary_metrics(counts):
    npred = counts["predicted_boundary_pixels"]
    ntrue = counts["true_boundary_pixels"]
    precision = counts["matched_predicted_pixels"] / npred if npred else None
    recall = counts["matched_true_pixels"] / ntrue if ntrue else None

    if not npred and not ntrue:
        f1 = None
    elif precision is None or recall is None or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "boundary_f1": f1,
        "boundary_precision": precision,
        "boundary_recall": recall,
    }
