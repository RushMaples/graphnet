"""Independent density, gradient and optional spherical-flow checks."""

import importlib.util
import io
import math
import unittest

import numpy as np
from scipy.integrate import quad
import torch

from directional_heads import (KingDistribution, KingHead, SphericalFlowHead,
                               unit_vector, vmf_log_prob)


class DensityTests(unittest.TestCase):
    def test_king_normalizer_independent_adaptive_integration(self):
        # Independent split theta-domain integration, including narrow cores.
        for alpha in [1e-4, .003, .1, 2.0, 20.0]:
            for beta in [.05, .5, 1., 4., 100.]:
                dist = KingDistribution(torch.tensor([[0., 0., 1.]]),
                                        torch.tensor([alpha], dtype=torch.double),
                                        torch.tensor([beta], dtype=torch.double))
                logz = dist.log_normalizer.item()
                points = sorted(set([0., min(alpha, math.pi), min(10*alpha, math.pi), math.pi]))
                def integrand(theta):
                    return math.exp(-beta * math.log1p(theta**2/(2*beta*alpha**2)) - logz) * math.sin(theta) * 2*math.pi
                mass = sum(quad(integrand, a, b, epsabs=1e-10, epsrel=1e-10)[0]
                           for a, b in zip(points[:-1], points[1:]))
                self.assertAlmostEqual(mass, 1., places=7, msg=(alpha,beta))
                dist.check_quadrature()

    def test_cones_and_cdf(self):
        dist = KingDistribution(torch.tensor([[0.,0.,1.]]).repeat(3,1),
                                torch.tensor([1e-4,.1,10.]), torch.tensor([.5,4.,100.]))
        torch.testing.assert_close(dist.cdf(0), torch.zeros(3,dtype=torch.double))
        torch.testing.assert_close(dist.cdf(math.pi), torch.ones(3,dtype=torch.double))
        previous = torch.zeros(3,dtype=torch.double)
        for p in [.5,.68,.9,.95]:
            radius = dist.cone_radius(p)
            self.assertTrue((radius > previous).all())
            torch.testing.assert_close(dist.cdf(radius), torch.full((3,),p,dtype=torch.double), atol=1e-8,rtol=0)
            previous = radius

    def test_king_gradcheck(self):
        direction = torch.tensor([[.2,.3,.9]],dtype=torch.double,requires_grad=True)
        alpha = torch.tensor([.01],dtype=torch.double,requires_grad=True)
        beta = torch.tensor([2.],dtype=torch.double,requires_grad=True)
        truth = unit_vector(torch.tensor([[.4,.5,.8]],dtype=torch.double))
        self.assertTrue(torch.autograd.gradcheck(
            lambda d,a,b: KingDistribution(d,a,b).log_prob(truth), (direction,alpha,beta),
            eps=1e-6,atol=1e-4,rtol=1e-3))

    def test_head_context_and_count_gradients(self):
        torch.manual_seed(42)
        head=KingHead(context_dim=8)
        context=torch.randn(3,8,requires_grad=True)
        dist=head(context,torch.tensor([2,32,1000]))
        loss=-dist.log_prob(unit_vector(torch.randn(3,3))).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(context.grad).all())
        self.assertGreater(context.grad.abs().sum().item(),0)
        for param in head.parameters():
            self.assertTrue(torch.isfinite(param.grad).all())
        self.assertGreater(head.mapping[0].weight.grad[:,-1].abs().sum().item(),0)
        dist.check_quadrature()

    def test_vmf_uniform_and_effective_concentration(self):
        direction=torch.tensor([[0.,0.,.5]],dtype=torch.double)
        truth=torch.tensor([[0.,0.,1.]],dtype=torch.double)
        uniform=vmf_log_prob(direction,torch.tensor([0.]),truth)
        self.assertAlmostEqual(uniform.item(),-math.log(4*math.pi),places=12)
        result=vmf_log_prob(direction,torch.tensor([4.]),truth)
        expected=math.log(2)-math.log(4*math.pi)-math.log(math.sinh(2))+2
        self.assertAlmostEqual(result.item(),expected,places=12)

    def test_king_density_outside_autocast(self):
        head=KingHead(context_dim=8)
        context=torch.randn(2,8,requires_grad=True)
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            dist=head(context,torch.tensor([5,300]))
            loss=-dist.log_prob(unit_vector(torch.randn(2,3))).mean()
        self.assertEqual(loss.dtype,torch.float64)
        loss.backward()
        self.assertTrue(torch.isfinite(context.grad).all())


@unittest.skipUnless(importlib.util.find_spec('jammy_flows'), 'Optional jammy_flows not installed')
class FlowTests(unittest.TestCase):
    def test_flow_poles_gradients_and_checkpoint_reload(self):
        truth = unit_vector(torch.tensor([[0.,0.,1.], [0.,0.,-1.],
            [1.,0.,0.], [-1.,0.,0.], [0.,1.,0.], [0.,-1.,0.],
            [1e-9,-1e-9,1.], [-1e-9,1e-9,-1.]], dtype=torch.double))
        for depth in (4, 16):
            torch.manual_seed(42)
            head = SphericalFlowHead(context_dim=8, num_layers=depth)
            context = torch.randn(len(truth), 8, requires_grad=True)
            logp = head(context).log_prob(truth)
            self.assertTrue(torch.isfinite(logp).all())
            (-logp.mean()).backward()
            self.assertTrue(torch.isfinite(context.grad).all())
            for parameter in head.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            restored = SphericalFlowHead(context_dim=8, num_layers=depth)
            checkpoint = io.BytesIO()
            torch.save(head.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
            torch.testing.assert_close(restored(context.detach()).log_prob(truth), logp.detach())

    def test_flow_ordinary_density_matches_original_conversion(self):
        torch.manual_seed(42)
        patched = SphericalFlowHead(context_dim=8, num_layers=4)
        original = SphericalFlowHead(context_dim=8, num_layers=4)
        original.load_state_dict(patched.state_dict(), strict=True)
        for layer in original.pdf.modules():
            if "eucl_to_spherical_embedding" in layer.__dict__:
                delattr(layer, "eucl_to_spherical_embedding")
        context = torch.randn(64, 8)
        truth = unit_vector(torch.randn(64, 3, dtype=torch.double))
        torch.testing.assert_close(patched(context).log_prob(truth),
            original(context).log_prob(truth), atol=1e-7, rtol=1e-7)

    def test_flow_density_samples_and_gradients(self):
        torch.manual_seed(42)
        head=SphericalFlowHead(context_dim=8,num_layers=4)
        context=torch.randn(2,8,requires_grad=True)
        dist=head(context)
        truth=unit_vector(torch.randn(2,3))
        loss=-dist.log_prob(truth).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(context.grad).all())
        samples,logp=dist.sample_log_prob(8)
        self.assertEqual(samples.shape,(2,8,3))
        self.assertEqual(logp.shape,(2,8))
        torch.testing.assert_close(samples.norm(dim=-1),torch.ones(2,8,dtype=torch.double),atol=1e-8,rtol=1e-8)
        self.assertTrue(torch.isfinite(logp).all())
        rescored=torch.stack([dist.log_prob(samples[:,i]) for i in range(8)],dim=1)
        torch.testing.assert_close(logp,rescored,atol=1e-5,rtol=1e-5)


if __name__=='__main__':
    unittest.main()
