import unittest
import numpy as np
from src.metrics.mosaic_v1 import ProbabilityMosaic


class TestProbabilityMosaic(unittest.TestCase):
    def test_overlap_probability_mean_and_unique_count(self):
        m = ProbabilityMosaic(2, 3)
        y = np.zeros((2, 2), dtype=np.uint8)
        v = np.ones((2, 2), dtype=bool)
        m.add(np.full((2, 2), 0.2), y, v, 0, 0)
        m.add(np.full((2, 2), 0.8), y, v, 0, 1)
        p, _, valid = m.result()
        np.testing.assert_allclose(p, [[.2, .5, .8], [.2, .5, .8]])
        self.assertEqual(int(valid.sum()), 6)
        self.assertEqual(int(m.count.sum()), 8)

    def test_label_conflict_rejected_without_mutation(self):
        m = ProbabilityMosaic(2, 2)
        v = np.ones((2, 2), dtype=bool)
        m.add(np.zeros((2, 2)), np.zeros((2, 2), dtype=np.uint8), v, 0, 0)
        before = m.probability_sum.copy(), m.count.copy(), m.label.copy()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            m.add(np.ones((2, 2)), np.ones((2, 2), dtype=np.uint8), v, 0, 0)
        for old, new in zip(before, (m.probability_sum, m.count, m.label)):
            np.testing.assert_array_equal(old, new)

    def test_invalid_pixels_do_not_contribute(self):
        m = ProbabilityMosaic(1, 2)
        y = np.array([[1, 0]], dtype=np.uint8)
        m.add(np.array([[.9, .9]]), y, np.array([[True, False]]), 0, 0)
        m.add(
            np.array([[.1, .1]]), np.zeros((1, 2), dtype=np.uint8),
            np.array([[False, True]]), 0, 0,
        )
        p, truth, valid = m.result()
        np.testing.assert_allclose(p, [[.9, .1]])
        np.testing.assert_array_equal(truth, [[1, 0]])
        self.assertTrue(valid.all())

    def test_order_invariance_and_uncovered_pixels(self):
        outputs = []
        for values in ((.25, .75), (.75, .25)):
            m = ProbabilityMosaic(1, 2)
            for value in values:
                m.add(
                    np.array([[value]]), np.array([[1]], dtype=np.uint8),
                    np.array([[True]]), 0, 0,
                )
            outputs.append(m.result())
        for a, b in zip(*outputs):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(outputs[0][2][0, 1])

    def test_invalid_inputs(self):
        m = ProbabilityMosaic(1, 1)
        y = np.zeros((1, 1), dtype=np.uint8)
        v = np.ones((1, 1), dtype=bool)
        for probability in (np.array([[np.nan]]), np.array([[1.1]])):
            with self.assertRaises(ValueError):
                m.add(probability, y, v, 0, 0)
        with self.assertRaises(ValueError):
            m.add(np.zeros((1, 1)), y, v, 0, 1)


if __name__ == "__main__":
    unittest.main()
