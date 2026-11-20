import math, unittest
import torch
from src.training.segmentation_steps import masked_bce
class WeightedLossTests(unittest.TestCase):
    def test_weighted_and_masked(self):
        z=torch.tensor([[[[0.,0.,100.]]]],requires_grad=True)
        y=torch.tensor([[[[1.,0.,0.]]]])
        v=torch.tensor([[[[True,True,False]]]])
        loss=masked_bce(z,y,v,pos_weight=3.)
        self.assertAlmostEqual(float(loss.detach()),2*math.log(2),places=6)
        loss.backward()
        self.assertAlmostEqual(float(z.grad[0,0,0,0]),-.75,places=6)
        self.assertAlmostEqual(float(z.grad[0,0,0,1]),.25,places=6)
        self.assertEqual(float(z.grad[0,0,0,2]),0.)
    def test_default_and_invalid_weight(self):
        z=torch.zeros(1,1,1,1); y=torch.ones_like(z); v=torch.ones_like(z,dtype=torch.bool)
        self.assertTrue(torch.equal(masked_bce(z,y,v),masked_bce(z,y,v,pos_weight=1.)))
        for w in (0,-1,float('nan'),float('inf')):
            with self.subTest(weight=w), self.assertRaisesRegex(ValueError,'pos_weight'):
                masked_bce(z,y,v,pos_weight=w)
if __name__=='__main__': unittest.main()
