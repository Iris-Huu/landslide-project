import unittest
import numpy as np
from src.metrics.boundary_v1 import boundary_counts, boundary_metrics


class TestBoundary(unittest.TestCase):
    def setUp(self):
        self.valid = np.ones((40, 40), dtype=bool)
        self.truth = np.zeros((40, 40), dtype=np.uint8)
        self.truth[:, 20:] = 1

    def score(self, prediction, valid=None):
        return boundary_metrics(boundary_counts(
            prediction, self.truth,
            self.valid if valid is None else valid, tolerance=2
        ))

    def test_perfect_boundary(self):
        self.assertEqual(self.score(self.truth)["boundary_f1"], 1.0)

    def test_two_pixel_shift_within_tolerance(self):
        prediction = np.zeros_like(self.truth)
        prediction[:, 22:] = 1
        self.assertEqual(self.score(prediction)["boundary_f1"], 1.0)

    def test_three_pixel_shift_outside_tolerance(self):
        prediction = np.zeros_like(self.truth)
        prediction[:, 23:] = 1
        self.assertEqual(self.score(prediction)["boundary_f1"], 0.0)

    def test_empty_prediction(self):
        result = self.score(np.zeros_like(self.truth))
        self.assertEqual(result["boundary_f1"], 0.0)
        self.assertIsNone(result["boundary_precision"])
        self.assertEqual(result["boundary_recall"], 0.0)

    def test_no_scored_boundaries(self):
        empty = np.zeros_like(self.truth)
        result = boundary_metrics(boundary_counts(empty, empty, self.valid))
        self.assertIsNone(result["boundary_f1"])

    def test_invalid_region_does_not_create_scored_boundary(self):
        valid = self.valid.copy()
        valid[:, 16:25] = False
        result = self.score(self.truth, valid)
        self.assertIsNone(result["boundary_f1"])

    def test_all_invalid(self):
        result = self.score(self.truth, np.zeros_like(self.valid))
        self.assertIsNone(result["boundary_f1"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            boundary_counts(self.truth, self.truth, self.valid.astype(np.uint8))
        with self.assertRaises(ValueError):
            boundary_counts(self.truth, self.truth, self.valid, tolerance=-1)


if __name__ == "__main__":
    unittest.main()
