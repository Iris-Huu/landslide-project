"""CPU tests on generated data only; no real data access."""
import unittest
import torch
from src.models.unet import SmallUNet
from src.training.segmentation_steps import ExperimentConfig, masked_bce, metrics_from_logits, train_step, validation_step


torch.set_num_threads(1)


class UNetCPUTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig(base_channels=2, learning_rate=.01, seed=17, image_height=16, image_width=16)
        torch.manual_seed(self.cfg.seed)

    def test_shape_finite_and_dimensions(self):
        model = SmallUNet(self.cfg.base_channels)
        images = torch.rand(2, 3, self.cfg.image_height, self.cfg.image_width)
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))
        target = torch.zeros_like(output); valid = torch.ones_like(output, dtype=torch.bool)
        self.assertTrue(torch.isfinite(masked_bce(output, target, valid)).item())
        with self.assertRaisesRegex(ValueError, 'multiples of 4'): model(torch.zeros(1,3,15,16))

    def test_update_and_validation(self):
        model = SmallUNet(self.cfg.base_channels)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.learning_rate)
        image = torch.rand(1,3,16,16); target = torch.randint(0,2,(1,1,16,16)).float()
        valid = torch.ones_like(target,dtype=torch.bool)
        before = [p.detach().clone() for p in model.parameters()]
        loss = train_step(model, optimizer, image, target, valid)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertTrue(any(not torch.equal(p,b) for p,b in zip(model.parameters(),before)))
        result = validation_step(model, image, target, valid, self.cfg.threshold)
        self.assertEqual(result['threshold'], .5)
        self.assertEqual(result['confusion']['valid_pixels'], 256)

    def test_masked_loss_and_zero_valid(self):
        logits = torch.tensor([[[[0., 100.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0.]]]])
        valid = torch.tensor([[[[True, False]]]])
        loss = masked_bce(logits, target, valid)
        self.assertAlmostEqual(loss.item(), 0.69314718056, places=6)
        loss.backward()
        self.assertAlmostEqual(logits.grad[0,0,0,0].item(), -.5, places=6)
        self.assertEqual(logits.grad[0,0,0,1].item(), 0.)
        with self.assertRaisesRegex(ValueError, 'no valid pixels'):
            masked_bce(logits.detach(), target, torch.zeros_like(valid))

    def test_hand_calculated_validation(self):
        logits = torch.tensor([[[[2., 2., -2., -2., 5.]]]])
        target = torch.tensor([[[[1., 0., 1., 0., 0.]]]])
        valid = torch.tensor([[[[True, True, True, True, False]]]])
        result = metrics_from_logits(logits, target, valid, .5)
        counts = result['confusion']
        self.assertEqual(tuple(counts[k] for k in ('tp','fp','fn','tn')), (1,1,1,1))
        self.assertEqual(counts['metrics'], {'iou':1/3, 'f1':.5, 'precision':.5, 'recall':.5})

    def test_reproducibility(self):
        def run():
            torch.manual_seed(self.cfg.seed)
            model = SmallUNet(self.cfg.base_channels)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.learning_rate)
            image = torch.rand(1,3,16,16); target = torch.randint(0,2,(1,1,16,16)).float()
            valid = torch.ones_like(target, dtype=torch.bool)
            loss = train_step(model, optimizer, image, target, valid)
            return loss, [p.detach().clone() for p in model.parameters()]
        a, pa = run(); b, pb = run()
        self.assertEqual(a,b)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(pa,pb)))

    def test_invalid_inputs(self):
        logits = torch.zeros(1,1,2,2); target = torch.zeros_like(logits); valid = torch.ones_like(logits,dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,'boolean'): masked_bce(logits,target,valid.float())
        with self.assertRaisesRegex(ValueError,'0/1'): masked_bce(logits,target+2,valid)
        with self.assertRaisesRegex(ValueError,'matching'): masked_bce(logits,target[:,:,:1],valid)
        with self.assertRaisesRegex(ValueError,'threshold'): metrics_from_logits(logits,target,valid,1.5)

if __name__ == '__main__': unittest.main()
