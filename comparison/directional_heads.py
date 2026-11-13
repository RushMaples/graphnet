"""Spherical directional densities for the frozen-P14 comparison.

All log densities use solid angle (per steradian). The existing GraphNeT vMF
training branch stays unchanged; ``vmf_log_prob`` is its exact evaluator.
"""

from contextlib import nullcontext
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _no_autocast(value):
    if value.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=value.device.type, enabled=False)
    return nullcontext()


def unit_vector(value):
    """Normalize with a deterministic north-pole fallback for zero vectors."""
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    tiny = torch.finfo(value.dtype).tiny
    north = torch.zeros_like(value)
    north[..., 2] = 1
    return torch.where(norm > tiny, value / norm.clamp_min(tiny), north)


def angular_distance(left, right):
    left, right = unit_vector(left.double()), unit_vector(right.double())
    sine = torch.linalg.vector_norm(torch.linalg.cross(left, right, dim=-1), dim=-1)
    return torch.atan2(sine, (left * right).sum(-1))


def vmf_log_prob(direction, kappa, truth):
    """Evaluate natural vector kappa*direction without silently renormalizing it."""
    with _no_autocast(direction):
        natural = direction.double() * kappa.double().reshape(-1, 1)
        concentration = torch.linalg.vector_norm(natural, dim=-1)
        dot = (natural * unit_vector(truth.double())).sum(-1)
        safe = concentration.clamp_min(1e-4)
        regular = (safe.log() - math.log(2 * math.pi)
                   - torch.log(-torch.expm1(-2 * safe)) + dot - concentration)
        small = (-math.log(4 * math.pi) - concentration.square() / 6
                 + concentration.pow(4) / 180 + dot)
        return torch.where(concentration < 1e-4, small, regular)


@lru_cache(maxsize=8)
def _legendre(order):
    if order < 16:
        raise ValueError("At least 16 quadrature nodes are required")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


class KingDistribution:
    """Angle-squared King profile, normalized with the spherical sin(theta) measure.

    Quadrature in s=log1p(theta/alpha) resolves both narrow cores and broad tails.
    Density/integration arithmetic is FP64 and differentiable.
    """

    def __init__(self, direction, alpha, beta, quadrature_order=128):
        self.direction = unit_vector(direction.double())
        self.alpha, self.beta = alpha.double().reshape(-1), beta.double().reshape(-1)
        self.quadrature_order = int(quadrature_order)
        if self.direction.shape != (self.alpha.numel(), 3) or self.beta.shape != self.alpha.shape:
            raise ValueError("Expected direction [B,3], alpha [B], beta [B]")
        if not torch.isfinite(self.alpha).all() or not torch.isfinite(self.beta).all():
            raise ValueError("King width/tail must be finite")
        if (self.alpha <= 0).any() or (self.beta <= 0).any():
            raise ValueError("King width/tail must be positive")
        self._log_normalizer = None

    def _log_radial_integral(self, limit, order=None):
        with _no_autocast(self.alpha):
            limit = torch.as_tensor(limit, dtype=torch.float64, device=self.alpha.device)
            limit = torch.broadcast_to(limit, self.alpha.shape).clamp(0, math.pi)
            positive_limit = limit.clamp_min(torch.finfo(torch.float64).tiny)
            nodes, weights = _legendre(order or self.quadrature_order)
            nodes = torch.as_tensor(nodes, device=self.alpha.device)
            weights = torch.as_tensor(weights, device=self.alpha.device)
            bound = torch.log1p(positive_limit / self.alpha)
            s = bound[:, None] * nodes
            scaled_angle = torch.expm1(s)
            theta = self.alpha[:, None] * scaled_angle
            log_kernel = -self.beta[:, None] * torch.log1p(
                scaled_angle.square() / (2 * self.beta[:, None]))
            log_terms = (weights.log() + theta.sin().clamp_min(1e-300).log()
                         + self.alpha[:, None].log() + s + log_kernel)
            result = bound.log() + torch.logsumexp(log_terms, dim=-1)
            return torch.where(limit == 0, torch.full_like(result, -torch.inf), result)

    @property
    def log_normalizer(self):
        if self._log_normalizer is None:
            self._log_normalizer = math.log(2 * math.pi) + self._log_radial_integral(math.pi)
        return self._log_normalizer

    def log_prob(self, truth):
        with _no_autocast(self.alpha):
            angle = angular_distance(self.direction, truth)
            kernel = -self.beta * torch.log1p((angle / self.alpha).square() / (2 * self.beta))
            return kernel - self.log_normalizer

    def cdf(self, angle):
        log_mass = self._log_radial_integral(angle) - (self.log_normalizer - math.log(2 * math.pi))
        return log_mass.exp().clamp(0, 1)

    @torch.no_grad()
    def cone_radius(self, probability):
        if not 0 < float(probability) < 1:
            raise ValueError("Cone probability must be strictly between zero and one")
        low, high = torch.zeros_like(self.alpha), torch.full_like(self.alpha, math.pi)
        for _ in range(44):
            middle = (low + high) / 2
            below = self.cdf(middle) < probability
            low, high = torch.where(below, middle, low), torch.where(below, high, middle)
        return (low + high) / 2

    @torch.no_grad()
    def check_quadrature(self, atol=1e-7):
        difference = (self._log_radial_integral(math.pi, self.quadrature_order * 2)
                      - self._log_radial_integral(math.pi)).abs().max().item()
        if not math.isfinite(difference) or difference > atol:
            raise RuntimeError(f"King quadrature failed refinement: {difference:g}")
        return difference


class KingHead(nn.Module):
    """Learn direction, core width and tail from raw CLS and cleaned parent count."""

    def __init__(self, context_dim=384, hidden_dim=128, quadrature_order=128):
        super().__init__()
        self.quadrature_order = quadrature_order
        self.mapping = nn.Sequential(nn.Linear(context_dim + 1, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, 5))
        with torch.no_grad():
            self.mapping[-1].bias[3] = math.log(math.expm1(0.3 - 1e-4))
            self.mapping[-1].bias[4] = math.log(math.expm1(4.0 - 0.05))

    def forward(self, context, parent_count):
        count = parent_count.reshape(-1).to(device=context.device, dtype=torch.float32)
        if count.numel() != context.shape[0] or not torch.isfinite(count).all() or (count <= 0).any():
            raise ValueError("Expected one positive cleaned parent pulse count per event")
        conditional = torch.cat([context, (count / 32).log().to(context.dtype)[:, None]], dim=-1)
        raw = self.mapping(conditional)
        with _no_autocast(raw):
            raw = raw.double()
            alpha, beta = 1e-4 + F.softplus(raw[:, 3]), 0.05 + F.softplus(raw[:, 4])
            return KingDistribution(raw[:, :3], alpha, beta, self.quadrature_order)


def spherical_flow_options(num_layers):
    """Official TNF paper settings; adapt the final-layer override to pilot depth."""
    if num_layers not in {4, 16}:
        raise ValueError("Supported flow depths are 4 (prototype) and 16")
    common = {
        "add_vertical_rq_spline_flow": 1, "spline_num_basis_functions": -1,
        "vertical_smooth": 1, "vertical_flow_defs": "rr", "circular_flow_defs": "oo",
        "vertical_fix_boundary_derivative": 1, "min_kappa": 1e-10,
        "kappa_prediction": "direct_log_real_bounded", "kappa_clamping": 0,
        "vertical_restrict_max_min_width_height_ratio": -1.0,
        "vertical_fix_first_width_n_height_to_zero": 1,
        "vertical_independent_width_height_parametrization": 1,
        "add_circular_rq_spline_flow": 1, "circular_add_rotation": 0,
        "vertical_also_fix_second_width_to_zero": 1, "rotation_mode": "householder",
    }
    final = {"add_vertical_rq_spline_flow": 0, "add_circular_rq_spline_flow": 0,
             "add_rotation": 1, "min_kappa": 1e-10,
             "kappa_prediction": "direct_log_real_bounded", "rotation_mode": "householder"}
    return {"f": common, (0, num_layers - 1): {"f": final}}


class FlowDistribution:
    def __init__(self, pdf, context):
        self.pdf, self.context = pdf, context

    def log_prob(self, truth):
        with _no_autocast(self.context):
            logp, _, _ = self.pdf(unit_vector(truth.double()),
                                  conditional_input=self.context.double(),
                                  force_embedding_coordinates=True)
            return logp.reshape(-1)

    @torch.no_grad()
    def sample_log_prob(self, n):
        if int(n) < 1:
            raise ValueError("Sample count must be positive")
        with _no_autocast(self.context):
            context = self.context.double().repeat_interleave(int(n), dim=0)
            samples, _, logp, _ = self.pdf.sample(
                conditional_input=context, force_embedding_coordinates=True)
            return (samples.reshape(self.context.shape[0], int(n), 3),
                    logp.reshape(self.context.shape[0], int(n)))

    def sample(self, n):
        return self.sample_log_prob(n)[0]


class SphericalFlowHead(nn.Module):
    """Conditional S2 jammy_flows head; double arithmetic outside encoder AMP."""

    def __init__(self, context_dim=384, num_layers=4):
        super().__init__()
        try:
            import jammy_flows
        except ImportError as exc:
            raise ImportError("Flow arm requires jammy_flows==1.1.0 in the PACE environment") from exc
        self.num_layers = num_layers
        self.pdf = jammy_flows.pdf(
            "s2", "f" * num_layers, conditional_input_dim=context_dim,
            amortization_mlp_dims="128", options_overwrite=spherical_flow_options(num_layers)).double()
        _install_stable_s2_coordinates(self.pdf)

    def forward(self, context, parent_count=None):
        return FlowDistribution(self.pdf, context)


def _stable_s2_coordinates(self, x, log_det):
    """Stable S2 chart conversion; retain jammy's existing theta safeguard."""
    from jammy_flows.layers.spheres.sphere_base import return_safe_angle_within_pi
    rho = torch.linalg.vector_norm(x[..., :2], dim=-1, keepdim=True)
    theta = return_safe_angle_within_pi(torch.atan2(rho, x[..., 2:3]))
    pole = rho == 0
    # Azimuth is undefined at a pole. Set it to zero before atan2, avoiding
    # both NaN values and atan2(0,0)'s undefined backward derivatives.
    px = torch.where(pole, torch.ones_like(rho), x[..., 0:1])
    py = torch.where(pole, torch.zeros_like(rho), x[..., 1:2])
    phi = torch.remainder(torch.atan2(py, px), 2 * math.pi)
    return torch.cat((theta, phi), dim=-1), log_det - torch.log(torch.sin(theta)).sum(-1)


def _install_stable_s2_coordinates(pdf):
    """Apply the chart compatibility fix on instances, without changing packages."""
    from types import MethodType
    count = 0
    for layer in pdf.modules():
        if getattr(layer, "dimension", None) == 2 and callable(getattr(layer, "eucl_to_spherical_embedding", None)):
            layer.eucl_to_spherical_embedding = MethodType(_stable_s2_coordinates, layer)
            count += 1
    if not count:
        raise RuntimeError("No S2 layers found for stable coordinate conversion")
