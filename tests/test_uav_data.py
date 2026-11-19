import csv
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from src.data.uav_data import select_samples, read_sample, VALID_LIMITATION


class UavDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('img', 'mask'):
            (self.root / name).mkdir()
        self.manifest = self.root / 'manifest.csv'
        self.name = 'sample.tif'
        self.write_manifest([['UAV', '2', 'train_candidate', self.name]])
        self.image = np.array([[[0, 255], [128, 64]], [[10, 20], [30, 40]], [[255, 0], [1, 2]]], dtype=np.uint8)
        self.mask = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
        self.write_tiff('img', self.image)
        self.write_tiff('mask', self.mask)

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, rows, fields=('platform', 'component', 'split', 'filename')):
        with self.manifest.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows)

    def write_tiff(self, kind, array):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NotGeoreferencedWarning)
            with rasterio.open(self.root / kind / self.name, 'w', driver='GTiff', width=array.shape[2], height=array.shape[1], count=array.shape[0], dtype=array.dtype, transform=from_origin(10, 10, 1, 1)) as ds:
                ds.write(array)
                if kind == 'img' and array.shape[0] == 3:
                    ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

    def sample(self):
        return select_samples(self.manifest, 'train_candidate')[0]

    def test_valid_rgb_scaling_and_provenance(self):
        x = read_sample(self.sample(), self.root)
        np.testing.assert_allclose(x['image'], self.image.astype(np.float32) / 255)
        np.testing.assert_array_equal(x['mask'], self.mask[0])
        self.assertEqual((x['image'].shape, x['image'].dtype, x['mask'].dtype), ((3, 2, 2), np.dtype('float32'), np.dtype('uint8')))
        self.assertTrue(x['valid_mask'].all()); self.assertEqual(x['valid_mask'].dtype, np.bool_)
        self.assertEqual(x['metadata']['valid_limitation'], VALID_LIMITATION)
        self.assertEqual(x['metadata']['split'], 'train_candidate')
        self.assertEqual(len(x['metadata']['manifest_sha256']), 64)

    def test_dimensions_mismatch(self):
        self.write_tiff('mask', np.zeros((1, 3, 2), dtype='uint8'))
        with self.assertRaisesRegex(ValueError, 'sample.tif.*dimensions differ'): read_sample(self.sample(), self.root)

    def test_channels(self):
        for kind, data in [('img', np.zeros((1,2,2), dtype='uint8')), ('mask', np.zeros((3,2,2), dtype='uint8'))]:
            with self.subTest(kind=kind):
                self.write_tiff(kind, data)
                with self.assertRaisesRegex(ValueError, 'sample.tif.*channels'): read_sample(self.sample(), self.root)
                self.write_tiff(kind, self.image if kind == 'img' else self.mask)

    def test_dtype(self):
        for kind, data in [('img', self.image.astype('uint16')), ('mask', self.mask.astype('uint16'))]:
            with self.subTest(kind=kind):
                self.write_tiff(kind, data)
                with self.assertRaisesRegex(ValueError, 'sample.tif.*dtype'): read_sample(self.sample(), self.root)
                self.write_tiff(kind, self.image if kind == 'img' else self.mask)

    def test_bad_labels(self):
        self.write_tiff('mask', np.array([[[0, 2],[0, 1]]], dtype='uint8'))
        with self.assertRaisesRegex(ValueError, 'sample.tif.*labels must be 0/1'): read_sample(self.sample(), self.root)

    def test_missing_image_and_mask(self):
        for kind in ('img','mask'):
            with self.subTest(kind=kind):
                (self.root/kind/self.name).unlink()
                with self.assertRaisesRegex(ValueError, f'sample.tif.*missing {kind}'): read_sample(self.sample(), self.root)
                self.write_tiff(kind, self.image if kind == 'img' else self.mask)

    def test_duplicate_and_missing_field(self):
        row=['UAV','2','train_candidate',self.name]
        self.write_manifest([row,row])
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.sample()
        self.write_manifest([row[:3]], fields=('platform','component','split'))
        with self.assertRaisesRegex(ValueError, 'missing required'): self.sample()

    def test_disallowed_splits(self):
        for split in ('test_candidate', 'buffer', 'unassigned', 'other'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'Unsupported split'):
                select_samples(self.manifest, split)

    def test_invalid_validity_policy(self):
        with self.assertRaisesRegex(ValueError, 'unsupported valid-pixel policy'):
            read_sample(self.sample(), self.root, 'exclude_nodata')

    def test_invalid_rgb_interpretation(self):
        with rasterio.open(self.root/'img'/self.name, 'r+') as ds:
            ds.colorinterp = (ColorInterp.blue, ColorInterp.green, ColorInterp.red)
        with self.assertRaisesRegex(ValueError, 'RGB color interpretation'): read_sample(self.sample(), self.root)

if __name__ == '__main__': unittest.main()
