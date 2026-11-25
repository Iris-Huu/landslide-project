import unittest

import numpy as np

from src.data.nodata_v1 import valid_mask_v1


class TestNoDataV1(unittest.TestCase):
    def test_edge_black_is_ignored(self):
        image = np.full((3, 5, 5), 100, dtype=np.uint8)
        image[:, 0, 1] = 0
        image[:, 1, 1] = 0
        label = np.zeros((5, 5), dtype=np.uint8)

        valid = valid_mask_v1(image, label)

        self.assertFalse(valid[0, 1])
        self.assertFalse(valid[1, 1])
        self.assertTrue(valid[2, 2])

    def test_internal_black_is_retained(self):
        image = np.full((3, 5, 5), 100, dtype=np.uint8)
        image[:, 2, 2] = 0
        label = np.zeros((5, 5), dtype=np.uint8)

        self.assertTrue(valid_mask_v1(image, label)[2, 2])

    def test_white_is_retained(self):
        image = np.full((3, 4, 4), 255, dtype=np.uint8)
        label = np.zeros((4, 4), dtype=np.uint8)

        self.assertTrue(valid_mask_v1(image, label).all())

    def test_positive_label_is_retained(self):
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        label = np.zeros((4, 4), dtype=np.uint8)
        label[1, 1] = 1

        valid = valid_mask_v1(image, label)

        self.assertTrue(valid[1, 1])
        self.assertEqual(int(valid.sum()), 1)

    def test_all_black_background_is_ignored(self):
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        label = np.zeros((4, 4), dtype=np.uint8)

        self.assertFalse(valid_mask_v1(image, label).any())

    def test_diagonal_is_not_connected(self):
        image = np.full((3, 5, 5), 100, dtype=np.uint8)
        image[:, 0, 0] = 0
        image[:, 1, 1] = 0
        label = np.zeros((5, 5), dtype=np.uint8)

        valid = valid_mask_v1(image, label)

        self.assertFalse(valid[0, 0])
        self.assertTrue(valid[1, 1])

    def test_invalid_inputs(self):
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        label = np.zeros((4, 4), dtype=np.uint8)

        with self.assertRaises(ValueError):
            valid_mask_v1(image, np.zeros((3, 4), dtype=np.uint8))

        with self.assertRaises(ValueError):
            valid_mask_v1(image.astype(np.float32), label)

        label[0, 0] = 2
        with self.assertRaises(ValueError):
            valid_mask_v1(image, label)


if __name__ == "__main__":
    unittest.main()
