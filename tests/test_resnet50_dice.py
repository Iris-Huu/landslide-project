import tempfile,unittest
from pathlib import Path
import torch
from src.models.deeplabv3plus import build_deeplabv3plus
from src.training.segmentation_steps import train_step
from src.training.exploratory import evaluate,save_checkpoint,load_checkpoint

torch.set_num_threads(1)
class Resnet50DiceTests(unittest.TestCase):
 @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
 def test_gpu_resnet50_forward_backward_checkpoint(self):
    torch.manual_seed(17); model,weights=build_deeplabv3plus('resnet50'); model=model.to('cuda:0')
    self.assertEqual(weights['status'],'not_used_random_initialization')
    x=torch.rand(2,3,64,64,device='cuda:0'); y=torch.randint(0,2,(2,1,64,64),device='cuda:0').float(); v=torch.ones_like(y,dtype=torch.bool); v[:,:,:4,:4]=False
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001); model.eval()
    with torch.no_grad(): self.assertEqual(tuple(model(x).shape),(2,1,64,64))
    before=next(model.parameters()).detach().clone(); loss=train_step(model,optimizer,x,y,v,loss_kind='dice_bce')
    self.assertTrue(torch.isfinite(torch.tensor(loss))); self.assertFalse(torch.equal(before,next(model.parameters())))
    state={k:t.detach().clone() for k,t in model.state_dict().items()}; result=evaluate(model,[(x,y,v)],'cuda:0',loss_kind='dice_bce')
    self.assertEqual(result['confusion']['valid_pixels'],int(v.sum())); self.assertTrue(all(torch.equal(t,state[k]) for k,t in model.state_dict().items()))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'c.pt'; save_checkpoint(p,model,optimizer,1,{'name':'resnet50'},0.)
        restored,_=build_deeplabv3plus('resnet50'); restored=restored.to('cuda:0'); opt=torch.optim.Adam(restored.parameters(),lr=.0001)
        loaded=load_checkpoint(p,restored,opt,'cuda:0'); self.assertEqual(loaded['epoch'],1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())))
        self.assertEqual(len(optimizer.state),len(opt.state))
        for a,b in zip(optimizer.state.values(),opt.state.values()): self.assertTrue(all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a))
        self.assertTrue(torch.isfinite(torch.tensor(train_step(restored,opt,x,y,v,loss_kind='dice_bce'))))
if __name__=='__main__': unittest.main()
