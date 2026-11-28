"""Global spatial blocks and paired bootstrap for fixed segmentation models."""
import numpy as np

METRICS = ("iou", "f1", "precision", "recall")


def metrics(counts):
    c = np.asarray(counts, dtype=np.float64)
    tp, fp, fn, tn = np.moveaxis(c, -1, 0)
    numerator = np.stack((tp, 2*tp, tp, tp), axis=-1)
    denominator = np.stack(
        (tp+fp+fn, 2*tp+fp+fn, tp+fp, tp+fn), axis=-1
    )
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def block_counts(prediction, truth, valid, top, left, side=833):
    p, y, v = map(np.asarray, (prediction, truth, valid))
    if p.ndim != 2 or p.shape != y.shape or p.shape != v.shape:
        raise ValueError("Matching two-dimensional arrays required")
    if v.dtype != np.bool_:
        raise ValueError("Boolean valid mask required")
    if not np.isin(p, (0, 1)).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Binary prediction and truth required")
    if not isinstance(side, int) or side <= 0:
        raise ValueError("Positive integer block side required")
    if not all(isinstance(z, (int, np.integer)) and z >= 0 for z in (top, left)):
        raise ValueError("Nonnegative integer origin required")
    h, w = y.shape
    if not h or not w:
        raise ValueError("Empty arrays")

    ids, counts = [], []
    for by in range(top // side, (top+h-1) // side + 1):
        for bx in range(left // side, (left+w-1) // side + 1):
            region = np.s_[
                max(0, by*side-top):min(h, (by+1)*side-top),
                max(0, bx*side-left):min(w, (bx+1)*side-left),
            ]
            mask = v[region]
            if not mask.any():
                continue
            a = p[region][mask].astype(bool)
            b = y[region][mask].astype(bool)
            ids.append([by, bx])
            counts.append([
                int((a & b).sum()), int((a & ~b).sum()),
                int((~a & b).sum()), int((~a & ~b).sum()),
            ])
    if not counts:
        raise ValueError("No valid spatial blocks")
    return np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.int64)


def bootstrap(counts, replicates=10000, seed=20261127):
    c = np.asarray(counts)
    if c.ndim != 3 or c.shape[2] != 4 or min(c.shape[:2]) < 1:
        raise ValueError("Expected models x blocks x 4 counts")
    if not np.issubdtype(c.dtype, np.integer) or (c < 0).any():
        raise ValueError("Nonnegative integer counts required")
    if (c.sum(axis=2) == 0).any():
        raise ValueError("Every block must contain valid pixels")

    positive = c[:, :, 0] + c[:, :, 2]
    negative = c[:, :, 1] + c[:, :, 3]
    if not (
        (positive == positive[0]).all()
        and (negative == negative[0]).all()
    ):
        raise ValueError("Models have inconsistent block truth or validity")
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError("Positive replicate count required")

    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, c.shape[0], 4), dtype=np.float64)
    for i in range(replicates):
        draw = rng.integers(0, c.shape[1], size=c.shape[1])
        samples[i] = metrics(c[:, draw, :].sum(axis=1))

    def interval(values):
        answer = {}
        for index, name in enumerate(METRICS):
            column = values[:, index]
            finite = column[np.isfinite(column)]
            answer[name] = {
                "percentile_95": (
                    np.percentile(finite, [2.5, 97.5]).tolist()
                    if len(finite) else None
                ),
                "defined_replicates": int(len(finite)),
                "undefined_replicates": int(replicates-len(finite)),
            }
        return answer

    paired_mean = np.full((replicates, 4), np.nan)
    complete = np.isfinite(samples).all(axis=1)
    np.divide(
        np.where(np.isfinite(samples), samples, 0).sum(axis=1),
        c.shape[0], out=paired_mean, where=complete,
    )
    return {
        "replicates": replicates,
        "seed": seed,
        "block_count": int(c.shape[1]),
        "per_seed": [interval(samples[:, s]) for s in range(c.shape[0])],
        "paired_seed_mean": interval(paired_mean),
        "limitation": "Conditional on fixed models and this test region.",
    }
