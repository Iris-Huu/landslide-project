import unittest
import numpy as np
from src.data_audit.audit_nodata_candidate import find_candidate
from src.metrics.binary_segmentation import ConfusionMatrix
from scripts.splits.review_nodata_supplement_v1 import validate_partition


class ReviewTest(unittest.TestCase):
    def test_edge_connectivity_and_label_overlap(self):
        rgb = np.full((3, 5, 5), 120, np.uint8)
        rgb[:, 0, 0:2] = 0
        rgb[:, 1, 1] = 0
        rgb[:, 3, 3] = 255
        got = find_candidate(rgb)
        want = np.zeros((5, 5), bool)
        want[0, :2] = True
        want[1, 1] = True
        np.testing.assert_array_equal(got, want)
        label = np.zeros((5, 5), bool)
        label[1, 1] = True
        self.assertEqual(int((label & got).sum()), 1)

    def test_confusion_partition_and_error(self):
        pred = np.array([1, 1, 0, 0], bool)
        truth = np.array([1, 0, 1, 0], bool)
        valid = np.array([1, 0, 1, 1], bool)
        full = ConfusionMatrix().update(pred, truth).result()
        keep = ConfusionMatrix().update(pred, truth, valid).result()
        self.assertEqual(validate_partition(full, keep, {"total": 1, "positive": 0, "fp": 1}),
                         {"tp": 0, "fp": 1, "fn": 0, "tn": 0})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_partition(full, keep, {"total": 1, "positive": 1, "fp": 1})


if __name__ == "__main__":
    unittest.main()
