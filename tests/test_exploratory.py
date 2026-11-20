import tempfile, unittest
from pathlib import Path
import torch
from src.models.unet import SmallUNet
from src.training.segmentation_steps import train_step
from src.training.exploratory import CandidateDataset,evaluate,save_checkpoint,load_checkpoint

torch.set_num_threads(1)
class ExploratoryTests(unittest.TestCase):
    def test_refused_split(self):
        for split in ('test_candidate','buffer','unassigned'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError,'only'):
                CandidateDataset('missing.csv','missing',split)
    def test_validation_state_and_mask(self):
        model=SmallUNet(2); model.train(); before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.rand(1,3,16,16); y=torch.zeros(1,1,16,16); v=torch.zeros(1,1,16,16,dtype=torch.bool); v[:,:,:4,:4]=True
        r=evaluate(model,[(x,y,v)],'cpu'); self.assertEqual(r['confusion']['valid_pixels'],16)
        self.assertTrue(model.training); self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_checkpoint_resume_state_rng(self):
        torch.manual_seed(17); m=SmallUNet(2); opt=torch.optim.Adam(m.parameters(),lr=.001)
        x=torch.rand(1,3,16,16); y=torch.zeros(1,1,16,16); v=torch.ones(1,1,16,16,dtype=torch.bool)
        train_step(m,opt,x,y,v)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt'; save_checkpoint(p,m,opt,1,{'test':True},.4)
            expected=torch.rand(4); n=SmallUNet(2); other=torch.optim.Adam(n.parameters(),lr=.001)
            state=load_checkpoint(p,n,other,'cpu'); self.assertEqual(state['epoch'],1); self.assertEqual(state['best'],.4)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),n.parameters())))
            self.assertEqual(len(opt.state),len(other.state)); self.assertTrue(torch.equal(expected,torch.rand(4)))
            for old,new in zip(opt.state.values(),other.state.values()):
                self.assertTrue(all(torch.equal(old[k],new[k]) for k in old))
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_checkpoint_rng_restore(self):
        torch.manual_seed(19); torch.cuda.manual_seed_all(19)
        model=SmallUNet(2).to('cuda'); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt'; save_checkpoint(p,model,optimizer,0,{},0.)
            expected=torch.rand(4,device='cuda')
            other=SmallUNet(2).to('cuda'); opt=torch.optim.Adam(other.parameters(),lr=.001)
            load_checkpoint(p,other,opt,'cuda')
            self.assertTrue(torch.equal(expected,torch.rand(4,device='cuda')))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters())))

if __name__=='__main__': unittest.main()
