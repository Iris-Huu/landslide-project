import unittest
import numpy as np
from src.metrics.region_analysis import center_mask, regional_counts, calibration_bins
from src.training.exploratory import CandidateDataset, evaluate
from src.models.unet import SmallUNet
import torch

class RegionTests(unittest.TestCase):
    def test_partition_and_empty_metrics(self):
        p=np.array([[1,0,1,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]],dtype=bool)
        t=np.array([[1,1,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],dtype=bool)
        c=center_mask((4,4),2); v=np.ones_like(c)
        r=regional_counts(p,t,c,v)
        self.assertEqual(tuple(r['full'][k] for k in ('tp','fp','fn','tn')),(2,2,1,11))
        for k in ('tp','fp','fn','tn'): self.assertEqual(r['full'][k],r['center'][k]+r['outside'][k])
        self.assertIsNone(regional_counts(np.zeros((2,2),bool),np.zeros((2,2),bool),center_mask((2,2),2),np.ones((2,2),bool))['full']['metrics']['precision'])
    def test_bins_boundaries(self):
        p=np.array([0.,.1,.5,1.]); y=np.array([0,1,1,0]); valid=np.array([True,True,True,True])
        bins=calibration_bins(p,y,valid); self.assertEqual([bins[i]['count'] for i in (0,1,5,9)],[1,1,1,1]); self.assertEqual(sum(b['positive_pixels'] for b in bins),2)
    def test_rejected_splits_and_eval(self):
        for name in ('test_candidate','buffer','unassigned'):
            with self.assertRaises(ValueError): CandidateDataset('unused','unused',name)
        m=SmallUNet(2); m.train(); before={k:v.clone() for k,v in m.state_dict().items()}
        evaluate(m,[(torch.zeros(1,3,8,8),torch.zeros(1,1,8,8),torch.ones(1,1,8,8,dtype=torch.bool))],'cpu')
        self.assertTrue(m.training); self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))
if __name__=='__main__': unittest.main()
