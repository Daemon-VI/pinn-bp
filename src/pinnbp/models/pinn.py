"""The physics-informed model.

Structure, and the reasoning behind it::

    PPG window (3 x L)
        |
        v
    PPGEncoder  ->  latent z (128)
        |                    |
        |                    +--> ParameterHead -> R, C, SV, HR, Tsys, Zc
        |                                              (bounded, physiological units)
        v
    PressureField(z, t) -> P(t)  in mmHg
        |
        +--> dense evaluation over one cardiac cycle -> SBP = max P, DBP = min P

The model never regresses SBP and DBP directly. It predicts a *pressure waveform*, and the
labels are read off that waveform the same way a clinician reads them off an arterial line.
This is what gives the physics somewhere to act: the Windkessel residual constrains P(t)
and its time derivative, and SBP/DBP inherit that constraint rather than being free
parameters.

Two structural choices carry constraints that would otherwise have to be learned:

**Bounded parameters.** The parameter head emits values through a scaled sigmoid into
generous physiological ranges, so R can never be negative and compliance can never be
zero -- a hard guarantee, not a penalty the optimiser might trade away. The ranges are set
*wider* than the hinge bands in :func:`pinnbp.physics.losses.physiological_prior`, so the
soft prior still does real work near the extremes while the hard bound rules out the
impossible.

**Harmonic time basis.** The field takes cardiac *phase* u = t/T and expands it in integer
harmonics, so P(t) is periodic with period T by construction. Steady-state arterial pressure
is periodic, and building that in is strictly better than penalising its violation. It does
make the ``periodicity`` loss term near-vacuous under the default basis -- that term exists
for the ``raw`` basis ablation, where it is load-bearing, and under ``harmonic`` it serves
as an assertion that should stay near zero. ``docs/RESULTS.md`` reports both.

The harmonic basis also band-limits the waveform, which is a real modelling decision and not
only a convenience: 8 harmonics of a ~1.2 Hz fundamental resolves content to ~10 Hz, which
covers the arterial pressure spectrum (>99% of its energy is below 10 Hz) while excluding
the high-frequency noise a free-form decoder would happily fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..physics.windkessel import WindkesselParams
from .encoder import PPGEncoder

__all__ = [
    "PINNBP",
    "ParameterHead",
    "PressureField",
    "PARAM_RANGES",
    "PARAM_INIT",
    "PINNOutput",
]


# Hard bounds for the parameter head. Deliberately wider than the hinge bands in
# ``physiological_prior`` so that the soft prior retains a role; these only exclude the
# physically impossible.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "R": (0.30, 3.00),     # mmHg*s/mL
    "C": (0.30, 4.50),     # mL/mmHg
    "SV": (20.0, 150.0),   # mL
    "HR": (35.0, 190.0),   # bpm
    "Tsys": (0.14, 0.48),  # s
    "Zc": (0.010, 0.150),  # mmHg*s/mL
}
_PARAM_ORDER = ("R", "C", "SV", "HR", "Tsys", "Zc")

# Where an untrained network should start. These are healthy-adult typical values, NOT the
# midpoints of the ranges above -- the midpoint of the HR range is 112 bpm, which is
# tachycardic, and since the field's phase basis is scaled by the predicted period T=60/HR,
# starting there would put every subject's waveform on a badly wrong time axis for the
# first few epochs. Together these reproduce the 119/75 anchor in the windkessel module.
PARAM_INIT: dict[str, float] = {
    "R": 1.10,
    "C": 1.60,
    "SV": 70.0,
    "HR": 72.0,
    "Tsys": 0.30,
    "Zc": 0.055,
}


@dataclass
class PINNOutput:
    """Everything one forward pass produces.

    The waveform and the parameters are returned alongside SBP/DBP because they are the
    interpretable part of the model -- the thing the abstract calls a "clinically meaningful"
    prediction. A number with a compliance estimate and a pressure waveform behind it can be
    sanity-checked by a clinician; a bare number cannot.
    """

    sbp: Tensor
    dbp: Tensor
    map: Tensor
    params: WindkesselParams
    latent: Tensor
    waveform: Tensor
    t_dense: Tensor


class ParameterHead(nn.Module):
    """Map the latent to bounded cardiovascular parameters.

    The final bias is initialised by inverting the sigmoid at the typical values in
    :data:`PARAM_INIT`, so an untrained network predicts a healthy adult for everyone.
    Starting from a sane point rather than a random corner matters more than usual here:
    the Windkessel residual is only well conditioned for sane R and C, and a run that begins
    with a compliance near zero produces enormous gradients and diverges in the first few
    steps.
    """

    def __init__(self, latent_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(_PARAM_ORDER)),
        )
        # Small weights so the initial prediction is dominated by the bias and is nearly
        # identical for every subject; the network then has to earn its spread.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        with torch.no_grad():
            for i, name in enumerate(_PARAM_ORDER):
                lo, hi = PARAM_RANGES[name]
                frac = (PARAM_INIT[name] - lo) / (hi - lo)
                frac = min(max(frac, 1e-4), 1 - 1e-4)
                self.net[-1].bias[i] = float(np.log(frac / (1.0 - frac)))

    def forward(self, z: Tensor) -> WindkesselParams:
        raw = self.net(z)
        vals: dict[str, Tensor] = {}
        for i, name in enumerate(_PARAM_ORDER):
            lo, hi = PARAM_RANGES[name]
            vals[name] = lo + (hi - lo) * torch.sigmoid(raw[:, i])
        return WindkesselParams(
            R=vals["R"], C=vals["C"], SV=vals["SV"], HR=vals["HR"],
            Tsys=vals["Tsys"], Zc=vals["Zc"],
        )


class PressureField(nn.Module):
    """P(z, t): a continuous, differentiable pressure waveform conditioned on the latent.

    Continuous in t rather than a fixed vector of samples, which is what makes the
    collocation residual exact -- dP/dt comes from autograd, not from finite differences on
    a grid whose spacing would otherwise limit the accuracy of the physics.

    Args:
        latent_dim: encoder output width.
        n_harmonics: harmonics of the cardiac fundamental. 8 gives ~10 Hz bandwidth at a
            typical heart rate, which covers the arterial pressure spectrum.
        hidden: MLP width.
        basis: ``"harmonic"`` (periodic by construction, default) or ``"raw"`` (phase fed
            directly; periodicity must then be imposed by the loss). ``"raw"`` exists for
            the ablation that measures what the hard constraint is worth.
        p_centre, p_scale: output parameterisation, ``P = p_centre + p_scale * f(...)``.
            Centring on 90 mmHg means an untrained network emits a physiologically
            plausible flat pressure rather than noise around zero, so the physics terms are
            meaningful from the first step.
    """

    def __init__(
        self,
        latent_dim: int,
        n_harmonics: int = 8,
        hidden: int = 128,
        basis: str = "harmonic",
        p_centre: float = 90.0,
        p_scale: float = 40.0,
    ):
        super().__init__()
        if basis not in ("harmonic", "raw"):
            raise ValueError(f"unknown basis {basis!r}")
        self.basis = basis
        self.n_harmonics = n_harmonics
        self.p_centre = p_centre
        self.p_scale = p_scale

        n_time_feats = 2 * n_harmonics if basis == "harmonic" else 1
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_time_feats, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Small final weights: the field starts almost flat at p_centre and grows structure
        # as the data term pulls on it.
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def time_features(self, t: Tensor, period: Tensor) -> Tensor:
        """Expand time into the chosen basis.

        Args:
            t: ``(B, N)`` times in seconds.
            period: ``(B,)`` cardiac period in seconds.

        Returns:
            ``(B, N, F)`` features.
        """
        u = t / period.unsqueeze(1).clamp(min=1e-3)  # cardiac phase
        if self.basis == "raw":
            return u.unsqueeze(-1)
        k = torch.arange(1, self.n_harmonics + 1, device=t.device, dtype=t.dtype)
        ang = 2.0 * torch.pi * u.unsqueeze(-1) * k  # (B, N, K)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, z: Tensor, t: Tensor, period: Tensor) -> Tensor:
        """Evaluate the field.

        Args:
            z: ``(B, D)`` latent.
            t: ``(B, N)`` times in seconds.
            period: ``(B,)`` cardiac period.

        Returns:
            ``(B, N)`` pressure in mmHg.
        """
        feats = self.time_features(t, period)
        z_exp = z.unsqueeze(1).expand(-1, t.shape[1], -1)
        h = torch.cat([z_exp, feats], dim=-1)
        return self.p_centre + self.p_scale * self.net(h).squeeze(-1)


class PINNBP(nn.Module):
    """Physics-informed cuffless blood pressure model.

    Args:
        in_channels: input channels (PPG, VPG, APG).
        widths: encoder stage widths.
        latent_dim: encoder output width.
        n_harmonics: field harmonics.
        field_hidden: field MLP width.
        basis: field time basis; see :class:`PressureField`.
        n_dense: samples used to read SBP/DBP off the waveform. 128 over one cycle at
            ~0.85 s is ~7 ms resolution, which is finer than the systolic peak is sharp;
            below about 64 the sampled maximum starts to sit measurably below the true
            peak and SBP is biased low.
        dropout: encoder dropout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 48, 64, 96),
        latent_dim: int = 128,
        n_harmonics: int = 8,
        field_hidden: int = 128,
        basis: str = "harmonic",
        n_dense: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.encoder = PPGEncoder(in_channels, widths, latent_dim, dropout)
        self.param_head = ParameterHead(latent_dim)
        self.field = PressureField(latent_dim, n_harmonics, field_hidden, basis)
        self.n_dense = n_dense

    def field_fn(self, z: Tensor, t: Tensor, params: WindkesselParams):
        """Adapter matching the ``(latent, t) -> P`` signature the loss functions expect."""
        return self.field(z, t, params.period)

    def forward(self, x: Tensor, n_dense: int | None = None) -> PINNOutput:
        """Encode a batch of windows and read blood pressure off the predicted waveform.

        Args:
            x: ``(B, C, L)`` PPG channels.
            n_dense: override the dense sampling count.

        Returns:
            :class:`PINNOutput`.
        """
        z = self.encoder(x)
        params = self.param_head(z)

        n = n_dense or self.n_dense
        # Dense grid spanning exactly one cardiac cycle per subject. The endpoint is
        # excluded because with a periodic field P(T) == P(0), and including it would give
        # the phase-zero sample double weight in the MAP average.
        u = torch.linspace(0, 1, n + 1, device=x.device, dtype=x.dtype)[:-1]
        t = u.unsqueeze(0) * params.period.unsqueeze(1)

        waveform = self.field(z, t, params.period)

        sbp = waveform.amax(dim=1)
        dbp = waveform.amin(dim=1)
        map_ = waveform.mean(dim=1)

        return PINNOutput(
            sbp=sbp, dbp=dbp, map=map_, params=params, latent=z, waveform=waveform, t_dense=t
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
