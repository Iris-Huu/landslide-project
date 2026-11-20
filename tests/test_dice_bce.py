import unittest
import torch
from src.training.segmentation_steps import masked_dice_bce
class DiceBCETests(unittest.TestCase):
    def test_perfect_and_background(self):
        y=torch.tensor([[[[0.,1.],[1.,0.]]]]); z=(2*y-1)*25; valid=torch.ones_like(y,dtype=torch.bool)
        self.assertLess(masked_dice_bce(z,y,valid).item(),1e-6)
        self.assertLess(masked_dice_bce(torch.full_like(y,-25),torch.zeros_like(y),valid).item(),1e-6)
    def test_partial_mask_and_gradient(self):
        z=torch.tensor([[[[0.,100.]]]],requires_grad=True); y=torch.tensor([[[[1.,0.]]]]); valid=torch.tensor([[[[True,False]]]])
        loss=masked_dice_bce(z,y,valid); self.assertAlmostEqual(loss.item(),0.69314718056+0.2,places=6)
        loss.backward(); self.assertTrue(torch.isfinite(z.grad).all()); self.assertEqual(z.grad[0,0,0,1].item(),0.)
    def test_invalid(self):
        z=torch.zeros(1,1,2,2); y=torch.zeros_like(z); v=torch.ones_like(z,dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,'no valid'): masked_dice_bce(z,y,torch.zeros_like(v))
        with self.assertRaisesRegex(ValueError,'matching'): masked_dice_bce(z,y[:,:,:1],v)
        with self.assertRaisesRegex(ValueError,'positive'): masked_dice_bce(z,y,v,smooth=0)
        with self.assertRaisesRegex(ValueError,'weights'): masked_dice_bce(z,y,v,bce_weight=0,dice_weight=0)
if __name__=='__main__': unittest.main()
