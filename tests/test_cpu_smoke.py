"""End-to-end synthetic checks: no real sample access."""
import unittest
import numpy as np
import torch
from scripts.training.cpu_smoke import synthetic_gate, tensor_batch
from src.models.unet import SmallUNet
from src.training.segmentation_steps import train_step

class SmokeTests(unittest.TestCase):
    def test_end_to_end_record(self):
        torch.set_num_threads(1)
        record = synthetic_gate()
        self.assertEqual(record['mode'], 'synthetic')
        self.assertTrue(record['step']['parameter_changed'])
        self.assertEqual(record['step']['tensor_shapes']['image'], [1,3,16,16])
        self.assertEqual(record['config']['valid_policy'], 'all_pixels_provisional')

    def test_input_validation_and_identical_crop(self):
        image = np.zeros((3,20,20), dtype=np.float32)
        mask = np.zeros((20,20),dtype=np.uint8); valid = np.ones((20,20), dtype=bool)
        image[:,2:18,2:18] = 1; mask[2:18,2:18] = 1; valid[2:18,2:18] = False
        x,y,v,crop = tensor_batch({'image':image, 'mask':mask, 'valid_mask':valid},16)
        self.assertEqual(crop, {'top':2,'left':2,'height':16,'width':16})
        self.assertTrue(x.eq(1).all() and y.eq(1).all() and not v.any())
        with self.assertRaisesRegex(ValueError,'uint8'):
            tensor_batch({'image':image, 'mask':mask.astype(float), 'valid_mask':valid},16)

    def test_nonfinite_gradient_aborts_before_update(self):
        model=SmallUNet(2); optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        parameter=next(model.parameters()); before=parameter.detach().clone()
        handle=parameter.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        try:
            with self.assertRaisesRegex(ValueError, 'non-finite gradient'):
                train_step(model,optimizer,torch.rand(1,3,16,16),torch.zeros(1,1,16,16),torch.ones(1,1,16,16,dtype=torch.bool))
            self.assertTrue(torch.equal(parameter,before))
        finally:
            handle.remove()

if __name__ == '__main__': unittest.main()
