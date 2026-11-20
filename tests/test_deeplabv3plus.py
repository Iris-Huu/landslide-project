"""DeepLabV3+ synthetic checks; no real imagery."""
import tempfile, unittest
from pathlib import Path
import torch
from src.models.deeplabv3plus import build_deeplabv3plus
from src.training.segmentation_steps import train_step,masked_bce
from src.training.exploratory import evaluate,save_checkpoint,load_checkpoint,CandidateDataset

torch.set_num_threads(1)
class DeepLabTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17); self.model,self.weights=build_deeplabv3plus()
    def test_shape_gradient_update_and_mask(self):
        x=torch.rand(2,3,64,64); y=torch.randint(0,2,(2,1,64,64)).float(); valid=torch.ones_like(y,dtype=torch.bool)
        valid[:,:,:8,:8]=False; opt=torch.optim.Adam(self.model.parameters(),lr=.0001)
        z=self.model(x); self.assertEqual(tuple(z.shape),(2,1,64,64)); self.assertTrue(torch.isfinite(masked_bce(z,y,valid)))
        before=[p.detach().clone() for p in self.model.parameters()]; loss=train_step(self.model,opt,x,y,valid)
        self.assertTrue(torch.isfinite(torch.tensor(loss))); self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,self.model.parameters())))
        with self.assertRaisesRegex(ValueError,'no valid pixels'): masked_bce(z,y,torch.zeros_like(valid))
        self.assertEqual(self.weights['status'],'not_used_random_initialization')
    def test_evaluation_no_parameter_or_bn_change(self):
        self.model.train(); before={k:v.clone() for k,v in self.model.state_dict().items()}
        x=torch.rand(2,3,64,64); y=torch.zeros(2,1,64,64); v=torch.ones_like(y,dtype=torch.bool)
        r=evaluate(self.model,[(x,y,v)],'cpu'); self.assertEqual(r['confusion']['valid_pixels'],8192)
        self.assertTrue(self.model.training); self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
    def test_checkpoint_model_optimizer(self):
        x=torch.rand(2,3,64,64); y=torch.zeros(2,1,64,64); v=torch.ones_like(y,dtype=torch.bool)
        opt=torch.optim.Adam(self.model.parameters(),lr=.0001); train_step(self.model,opt,x,y,v)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt'; save_checkpoint(p,self.model,opt,1,{'encoder':'resnet18'},.25)
            model,_=build_deeplabv3plus(); other=torch.optim.Adam(model.parameters(),lr=.0001); ck=load_checkpoint(p,model,other,'cpu')
            self.assertEqual(ck['epoch'],1); self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.model.state_dict().values(),model.state_dict().values())))
            self.assertEqual(len(opt.state),len(other.state))
            for a,b in zip(opt.state.values(),other.state.values()):
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
    def test_invalid_and_split_guard(self):
        with self.assertRaises(Exception): self.model(torch.rand(2,2,64,64))
        with self.assertRaises(ValueError): build_deeplabv3plus(encoder='')
        with self.assertRaises(FileNotFoundError): build_deeplabv3plus(encoder_weights_path='/nonexistent/weights.pt')
        for split in ('test_candidate','buffer','unassigned'):
            with self.assertRaises(ValueError): CandidateDataset('missing','missing',split)
    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_gpu_synthetic_forward_backward(self):
        model=self.model.to('cuda'); x=torch.rand(2,3,64,64,device='cuda'); y=torch.zeros(2,1,64,64,device='cuda'); v=torch.ones_like(y,dtype=torch.bool)
        opt=torch.optim.SGD(model.parameters(),lr=.0001); before=next(model.parameters()).detach().clone()
        loss=train_step(model,opt,x,y,v); torch.cuda.synchronize(); self.assertTrue(torch.isfinite(torch.tensor(loss))); self.assertFalse(torch.equal(before,next(model.parameters())))
if __name__=='__main__': unittest.main()
