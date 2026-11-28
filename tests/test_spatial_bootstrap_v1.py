import unittest
import numpy as np
from src.metrics.spatial_bootstrap_v1 import block_counts, bootstrap, metrics


class TestSpatialBootstrap(unittest.TestCase):
    def test_known_metrics(self):
        np.testing.assert_allclose(metrics([1, 1, 1, 1]), [1/3, .5, .5, .5])

    def test_global_grid_origin_and_pixel_conservation(self):
        y = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        p = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        v = np.array([[True, True], [True, False]])
        ids, counts = block_counts(p, y, v, top=1, left=1, side=2)
        np.testing.assert_array_equal(ids, [[0, 0], [0, 1], [1, 0]])
        np.testing.assert_array_equal(counts.sum(axis=0), [1, 1, 1, 0])

    def test_default_10000_and_undefined_precision(self):
        result = bootstrap(np.array([[[0, 0, 2, 3]]], dtype=np.int64))
        self.assertEqual(result["replicates"], 10000)
        self.assertEqual(result["per_seed"][0]["iou"]["percentile_95"], [0., 0.])
        self.assertIsNone(result["per_seed"][0]["precision"]["percentile_95"])
        self.assertEqual(
            result["per_seed"][0]["precision"]["undefined_replicates"], 10000
        )

    def test_paired_draws_and_reproducibility(self):
        one = np.array([[2, 1, 1, 4], [1, 2, 3, 2]], dtype=np.int64)
        counts = np.stack((one, one))
        a = bootstrap(counts, replicates=50, seed=7)
        b = bootstrap(counts, replicates=50, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["per_seed"][0], a["paired_seed_mean"])

    def test_mean_of_metrics_not_pooling_models(self):
        counts = np.array([[[1, 0, 0, 1]], [[1, 1, 0, 0]]], dtype=np.int64)
        result = bootstrap(counts, replicates=10)
        self.assertEqual(
            result["paired_seed_mean"]["iou"]["percentile_95"], [.75, .75]
        )

    def test_invalid_truth_and_empty_validity(self):
        counts = np.array([[[1, 0, 0, 1]], [[0, 0, 0, 2]]], dtype=np.int64)
        with self.assertRaises(ValueError):
            bootstrap(counts)
        with self.assertRaises(ValueError):
            block_counts(
                np.zeros((1, 1)), np.zeros((1, 1)),
                np.zeros((1, 1), dtype=bool), 0, 0,
            )


if __name__ == "__main__":
    unittest.main()
