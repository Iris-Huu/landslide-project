import tempfile, unittest
from pathlib import Path
import torch
from src.models.segformer import build_segformer
from src.training.segmentation_steps import masked_dice_bce,train_step
from src.training.exploratory import CandidateDataset,evaluate,save_checkpoint,load_checkpoint

torch.set_num_threads(1)
class SegformerTests(unittest.TestCase):
 @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
 def test_gpu_train_eval_and_checkpoint(self):
    torch.manual_seed(17); model,weights=build_segformer('mit_b2'); model=model.to('cuda:0'); self.assertEqual(weights['status'],'not_used_random_initialization')
    x=torch.rand(2,3,128,128,device='cuda'); y=torch.randint(0,2,(2,1,128,128),device='cuda').float(); v=torch.ones_like(y,dtype=torch.bool)
    model.eval()
    with torch.no_grad(): z=model(x)
    self.assertEqual(tuple(z.shape),(2,1,128,128)); self.assertTrue(torch.isfinite(masked_dice_bce(z,y,v)))
    with self.assertRaisesRegex(ValueError,'no valid'): masked_dice_bce(z,y,torch.zeros_like(v))
    opt=torch.optim.Adam(model.parameters(),lr=.0001); before=next(model.parameters()).detach().clone()
    self.assertTrue(torch.isfinite(torch.tensor(train_step(model,opt,x,y,v,loss_kind='dice_bce')))); self.assertFalse(torch.equal(before,next(model.parameters())))
    state={k:t.clone() for k,t in model.state_dict().items()}; r=evaluate(model,[(x,y,v)],'cuda:0',loss_kind='dice_bce')
    self.assertTrue(model.training); self.assertEqual(r['confusion']['valid_pixels'],32768); self.assertTrue(all(torch.equal(t,state[k]) for k,t in model.state_dict().items()))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'model.pt'; save_checkpoint(p,model,opt,1,{'arch':'mit_b2'},0.)
        restored,_=build_segformer(); restored=restored.to('cuda:0'); other=torch.optim.Adam(restored.parameters(),lr=.0001); cp=load_checkpoint(p,restored,other,'cuda:0')
        self.assertEqual(cp['epoch'],1); self.assertEqual(len(other.state),len(opt.state))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())))
        for a,b in zip(opt.state.values(),other.state.values()): self.assertTrue(all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a))
 def test_restrictions(self):
    for split in ('test_candidate','buffer','unassigned'):
        with self.assertRaises(ValueError): CandidateDataset('unused','unused',split)
    with self.assertRaises(ValueError): build_segformer('')
if __name__=='__main__': unittest.main()
