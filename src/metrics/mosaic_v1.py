"""Probability mosaic with overlap deduplication and label conflict checks."""
import numpy as np


class ProbabilityMosaic:
    def __init__(self, height, width):
        if height <= 0 or width <= 0:
            raise ValueError("Invalid mosaic shape")
        self.probability_sum = np.zeros((height, width), dtype=np.float64)
        self.count = np.zeros((height, width), dtype=np.uint32)
        self.label = np.zeros((height, width), dtype=np.uint8)

    def add(self, probability, label, valid, top, left):
        probability = np.asarray(probability)
        label = np.asarray(label)
        valid = np.asarray(valid)
        if probability.ndim != 2 or (
            probability.shape != label.shape or label.shape != valid.shape
        ):
            raise ValueError("Matching two-dimensional arrays required")
        if valid.dtype != np.bool_:
            raise ValueError("Boolean valid mask required")
        if not np.isfinite(probability).all() or (
            (probability < 0) | (probability > 1)
        ).any():
            raise ValueError("Probabilities must be finite and within [0,1]")
        if not np.isin(label, (0, 1)).all():
            raise ValueError("Labels must be 0/1")

        h, w = label.shape
        H, W = self.count.shape
        if top < 0 or left < 0 or top+h > H or left+w > W:
            raise ValueError("Tile outside mosaic")

        region = np.s_[top:top+h, left:left+w]
        counts = self.count[region]
        labels = self.label[region]
        conflict = valid & (counts > 0) & (labels != label)
        if conflict.any():
            raise ValueError(
                f"Conflicting valid overlap labels: {int(conflict.sum())}"
            )

        # Validate before mutation, so rejected tiles leave state unchanged.
        sums = self.probability_sum[region]
        sums[valid] += probability[valid]
        labels[valid] = label[valid]
        counts[valid] += 1

    def result(self):
        valid = self.count > 0
        probability = np.zeros(self.count.shape, dtype=np.float64)
        np.divide(
            self.probability_sum, self.count,
            out=probability, where=valid,
        )
        return probability, self.label.copy(), valid
