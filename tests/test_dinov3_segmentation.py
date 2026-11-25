import tempfile
import unittest
from pathlib import Path
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from src.models.dinov3_segmentation import FrozenDINOv3Segmentation
from src.training.segmentation_steps import train_step, masked_dice_bce
from src.training.exploratory import evaluate, save_checkpoint, load_checkpoint, CandidateDataset


class DINOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        cfg = DINOv3ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=64, patch_size=16, num_register_tokens=4, image_size=32)
        self.model = FrozenDINOv3Segmentation(DINOv3ViTModel(cfg), [.4,.4,.4], [.2,.2,.2])
        self.x = torch.rand(2,3,32,32)
        self.y = torch.randint(0,2,(2,1,32,32)).float()
        self.v = torch.ones_like(self.y,dtype=torch.bool)

    def test_shape_update_frozen_eval_and_restore(self):
        m=self.model
        initial={k:v.clone() for k,v in m.encoder.state_dict().items()}
        before=m.head.weight.detach().clone()
        opt=torch.optim.Adam(m.head.parameters(),lr=.001)
        loss=train_step(m,opt,self.x,self.y,self.v,loss_kind='dice_bce')
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(tuple(m(self.x).shape),(2,1,32,32))
        self.assertFalse(torch.equal(before,m.head.weight))
        self.assertFalse(m.encoder.training)
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in m.encoder.state_dict().items()))
        state={k:v.clone() for k,v in m.state_dict().items()}
        evaluate(m,[(self.x,self.y,self.v)],torch.device('cpu'),loss_kind='dice_bce')
        self.assertTrue(all(torch.equal(state[k],v) for k,v in m.state_dict().items()))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt'; save_checkpoint(p,m,opt,1,{'kind':'synthetic'},.1)
            with torch.no_grad(): m.head.weight.add_(1)
            load_checkpoint(p,m,opt,torch.device('cpu'))
            self.assertTrue(all(torch.equal(state[k],v) for k,v in m.state_dict().items()))

    def test_invalid_and_mask(self):
        with self.assertRaises(ValueError): self.model(torch.rand(1,3,31,32))
        with self.assertRaises(ValueError): self.model(torch.rand(1,3,32,32)*2)
        with self.assertRaises(ValueError): masked_dice_bce(self.model(self.x),self.y,torch.zeros_like(self.v))
        with self.assertRaises(ValueError): CandidateDataset(None,None,'test_candidate')

if __name__ == '__main__': unittest.main()
