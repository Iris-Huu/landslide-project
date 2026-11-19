import json
import unittest
import numpy as np
from src.metrics.binary_segmentation import ConfusionMatrix


class BinarySegmentationTests(unittest.TestCase):
    def check_case(self, prediction, target, expected_counts, expected_metrics, valid=None):
        r = ConfusionMatrix().update(prediction, target, valid).result()
        self.assertEqual(tuple(r[k] for k in ('tp', 'fp', 'fn', 'tn')), expected_counts)
        self.assertEqual(r['metrics'], expected_metrics)
        json.dumps(r, allow_nan=False)
        return r

    def test_perfect(self):
        self.check_case([1, 0], [1, 0], (1, 0, 0, 1), dict(iou=1, f1=1, precision=1, recall=1))

    def test_fp_fn(self):
        self.check_case([1, 1, 0, 0], [1, 0, 1, 0], (1, 1, 1, 1), dict(iou=1/3, f1=.5, precision=.5, recall=.5))

    def test_both_empty(self):
        self.check_case([0, 0], [0, 0], (0, 0, 0, 2), dict(iou=None, f1=None, precision=None, recall=None))

    def test_truth_only(self):
        self.check_case([0, 0], [1, 0], (0, 0, 1, 1), dict(iou=0, f1=0, precision=None, recall=0))

    def test_prediction_only(self):
        self.check_case([1, 0], [0, 0], (0, 1, 0, 1), dict(iou=0, f1=0, precision=0, recall=None))

    def test_partial_valid(self):
        self.check_case([1, 1, 0, 0], [1, 0, 1, 0], (1, 0, 1, 0), dict(iou=.5, f1=2/3, precision=1, recall=.5), np.array([True, False, True, False]))

    def test_no_valid_pixels(self):
        r = self.check_case([1, 0], [1, 0], (0, 0, 0, 0), dict(iou=None, f1=None, precision=None, recall=None), np.array([False, False]))
        self.assertEqual((r['valid_pixels'], r['empty_samples']), (0, 1))

    def test_invalid_inputs(self):
        for pred, target, valid in [([0], [0, 1], None), ([0], [2], None), ([float('nan')], [0], None), ([0], [0], [1]), ([0], [0], np.array([True, False]))]:
            with self.subTest(pred=pred, target=target, valid=valid), self.assertRaises(ValueError):
                ConfusionMatrix().update(pred, target, valid)

    def test_pooled_is_not_per_image_mean(self):
        first = ConfusionMatrix().update([1], [1]).result()
        second = ConfusionMatrix().update([0, 0, 0], [1, 1, 1]).result()
        pooled = ConfusionMatrix().update([1], [1]).update([0, 0, 0], [1, 1, 1]).result()
        self.assertEqual((pooled['tp'], pooled['fn'], pooled['samples']), (1, 3, 2))
        self.assertEqual(pooled['metrics']['iou'], .25)
        self.assertEqual((first['metrics']['iou'] + second['metrics']['iou']) / 2, .5)

if __name__ == '__main__':
    unittest.main()
