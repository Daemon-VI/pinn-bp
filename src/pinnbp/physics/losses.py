"""Physics-informed loss terms.

What separates this from "a CNN with a regulariser bolted on" is that the network does not
predict two numbers -- it predicts a *pressure field* P(t) and a set of *physical
parameters* (R, C, SV, Tsys), and these losses force the two to agree with the Windkessel
ODE that relates them. SBP and DBP are then read off the field rather than regressed
directly.

The terms, and why each one earns its place:

``ode_residual``
    The actual PINN term. Samples collocation times inside the cardiac cycle, takes dP/dt
    by autograd, and penalises C dP/dt + P/R - Q(t). Nothing about the decoder architecture
    stops it emitting an arbitrary squiggle; this is what pulls that squiggle onto the
    manifold of physically realisable pressure waveforms.

``periodicity``
    At steady state a cardiac cycle must close: P(0) = P(T), and the slopes must match too.
    Without it the residual alone admits a solution that spirals, because any exponential
    drift still satisfies the ODE for *some* initial condition.

``tau_consistency``
    Ties the network's R*C to the decay constant measured off the PPG diastolic limb, as a
    *rank* constraint standardised within the batch. This is the term that keeps R and C
    individually identifiable rather than only through their joint effect on the labels.
    It is deliberately scale-free: the measured value is an uncalibrated proxy roughly four
    times smaller than true R*C, so comparing the two absolutely would drive compliance
    badly wrong. See the function's docstring.

``map_consistency``
    The mean of the field must be close to DBP + (SBP-DBP)/3. A weak constraint, and
    deliberately weak -- the rule of thumb is itself approximate -- but it penalises
    waveforms with a physiologically implausible systolic/diastolic time balance.

``physiological_prior``
    One-sided hinges on parameter ranges and on SBP > DBP. Costs nothing when the network
    is behaving and prevents the degenerate solutions that a purely data-driven fit falls
    into when the input is pure artifact.

All terms return a scalar already reduced over the batch, so the trainer just weights and
sums them. Weights live in :class:`LossWeights`, and the defaults were chosen so that at
initialisation no single physics term exceeds the data term -- see ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .windkessel import ejection_flow, map_from_sbp_dbp

__all__ = [
    "LossWeights",
    "PhysicsLossOutput",
    "ode_residual",
    "periodicity",
    "tau_consistency",
    "map_consistency",
    "physiological_prior",
    "huber",
]


@dataclass
class LossWeights:
    """Relative weights of the loss terms.

    Defaults are not arbitrary. At initialisation the data term sits around 30-40 (mmHg,
    Huber) and each weighted physics term lands between 0.5 and 5, so physics shapes the
    solution without ever dominating the labels. Raising ``ode`` past about 1.0 reliably
    trades accuracy for smoothness -- the field gets very clean and the SBP error gets
    worse. That trade is measured in ``docs/RESULTS.md``.
    """

    data: float = 1.0
    ode: float = 0.10
    periodicity: float = 0.05
    tau: float = 0.05
    map_rule: float = 0.02
    prior: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PhysicsLossOutput:
    """Every term, kept separate so the trainer can log them individually.

    Watching these curves separately is the only practical way to tell "physics is helping"
    from "physics has collapsed the field to a constant", which look identical if you only
    watch the total.
    """

    total: Tensor
    data: Tensor
    ode: Tensor
    periodicity: Tensor
    tau: Tensor
    map_rule: Tensor
    prior: Tensor

    def items(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "data": float(self.data.detach()),
            "ode": float(self.ode.detach()),
            "per": float(self.periodicity.detach()),
            "tau": float(self.tau.detach()),
            "map": float(self.map_rule.detach()),
            "prior": float(self.prior.detach()),
        }


def huber(pred: Tensor, target: Tensor, delta: float = 5.0) -> Tensor:
    """Huber loss with a delta in mmHg.

    delta=5 is chosen to match the AAMI mean-error tolerance: errors inside the clinically
    acceptable band are penalised quadratically, and anything beyond it linearly, so a
    handful of badly corrupted windows cannot dominate the gradient the way they would
    under plain MSE.
    """
    return F.huber_loss(pred, target, delta=delta)


def ode_residual(
    field_fn,
    latent: Tensor,
    params,
    n_collocation: int = 64,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Windkessel collocation residual, normalised to be scale-free.

    Draws ``n_collocation`` times uniformly inside each subject's own cardiac cycle
    [0, T), evaluates the decoded pressure field there, and differentiates it with respect
    to time using autograd -- not finite differences. That distinction matters: the field
    is an MLP in t, so the derivative is exact, and the residual therefore measures the
    physics rather than the discretisation.

    Random collocation points rather than a fixed grid, resampled every step, stop the
    decoder from satisfying the ODE at a fixed set of positions and misbehaving between
    them.

    The residual C dP/dt + P/R - Q has units of mL/s and its natural scale grows with peak
    flow, so it is divided by the peak ejection flow of that subject before reduction.
    Otherwise subjects with a large stroke volume would silently receive a larger physics
    weight than subjects with a small one.

    Args:
        field_fn: callable ``(latent, t) -> P`` where ``t`` is ``(B, N)`` and the output is
            ``(B, N)`` in mmHg.
        latent: ``(B, D)`` encoder output.
        params: :class:`~pinnbp.physics.windkessel.WindkesselParams` predicted by the model.
        n_collocation: collocation points per subject per step.
        generator: optional RNG for reproducible collocation sampling.

    Returns:
        Scalar mean squared normalised residual.
    """
    B = latent.shape[0]
    T = params.period.unsqueeze(1)  # (B, 1)

    u = torch.rand(B, n_collocation, device=latent.device, dtype=latent.dtype, generator=generator)
    t = (u * T).requires_grad_(True)

    P = field_fn(latent, t)

    (dPdt,) = torch.autograd.grad(
        outputs=P,
        inputs=t,
        grad_outputs=torch.ones_like(P),
        create_graph=True,
    )

    Q = ejection_flow(t, params.SV.unsqueeze(1), params.Tsys.unsqueeze(1))
    R = params.R.unsqueeze(1)
    C = params.C.unsqueeze(1)

    residual = C * dPdt + P / R - Q

    # Peak half-sine flow for this subject, the natural scale of the residual.
    q_peak = torch.pi * params.SV / (2.0 * params.Tsys)
    residual = residual / q_peak.unsqueeze(1).clamp(min=1e-3)

    return residual.pow(2).mean()


def periodicity(field_fn, latent: Tensor, params) -> Tensor:
    """Closure of the cardiac cycle: P(0) = P(T) and dP/dt(0) = dP/dt(T).

    The ODE residual on its own is satisfied by any solution of the equation, including
    ones that have not settled onto a limit cycle. Real arterial pressure at rest is
    periodic beat to beat, so imposing closure removes that whole family of drifting
    solutions and, in practice, is what stops DBP wandering during early training.

    Matching the derivative as well as the value is what makes it C1-periodic; value-only
    matching leaves a visible kink at the beat boundary that shows up as a spurious
    high-frequency component in the residual.
    """
    B = latent.shape[0]
    T = params.period.unsqueeze(1)
    zero = torch.zeros(B, 1, device=latent.device, dtype=latent.dtype)

    t = torch.cat([zero, T], dim=1).requires_grad_(True)  # (B, 2)
    P = field_fn(latent, t)
    (dPdt,) = torch.autograd.grad(P, t, torch.ones_like(P), create_graph=True)

    value_gap = P[:, 0] - P[:, 1]
    slope_gap = dPdt[:, 0] - dPdt[:, 1]

    # Slope is in mmHg/s and runs an order of magnitude larger than the value gap, so it is
    # scaled down to keep the two halves of this term comparable.
    return value_gap.pow(2).mean() + 0.01 * slope_gap.pow(2).mean()


def tau_consistency(
    params, tau_measured: Tensor, valid: Tensor | None = None, min_valid: int = 8
) -> Tensor:
    """Tie predicted R*C to the decay constant measured from the signal -- in *rank*, not value.

    Both quantities are put in log space, then standardised to zero mean and unit variance
    across the valid entries of the batch before being compared. The loss therefore
    constrains only how subjects are *ordered and spread* by their decay constant, and is
    completely blind to any scale factor or offset between the two.

    That blindness is required, not a convenience. The decay constant recoverable from a
    band-passed PPG is not arterial R*C: on this project's simulator it correlates with the
    truth at r ~ 0.58 but sits about four times too low (median 0.56 s against 2.26 s),
    because the 0.5 Hz high-pass removes most of a 2-second exponential and the nonlinear
    volume-pressure curve distorts what remains. See ``pinnbp.dsp.features`` for the
    measurement. An earlier version of this function compared the two absolutely, in log
    space; that does not merely add noise, it actively drives predicted compliance toward a
    value four times too low, and it does so silently.

    So the measurement is used for exactly the information it carries -- monotone ordering
    -- and nothing more.

    Args:
        params: predicted parameters.
        tau_measured: ``(B,)`` decay constants in seconds from
            :func:`pinnbp.dsp.features.diastolic_tau`. Uncalibrated.
        valid: optional ``(B,)`` boolean mask. Windows whose fit was poor must be excluded;
            a tau measured off a motion artifact is worse than no tau at all.
        min_valid: below this many valid entries the batch statistics are too unstable to
            standardise against, and the term returns zero rather than a noisy gradient.

    Returns:
        Scalar mean squared difference of standardised log values, or zero when the batch
        carries too few valid measurements.
    """
    pred = torch.log(params.tau.clamp(min=1e-3))
    meas = torch.log(tau_measured.clamp(min=1e-3))

    if valid is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    else:
        mask = valid.bool()

    n = int(mask.sum())
    if n < min_valid:
        return pred.sum() * 0.0  # keeps the graph connected, contributes nothing

    p_v, m_v = pred[mask], meas[mask]
    p_z = (p_v - p_v.mean()) / (p_v.std() + 1e-6)
    m_z = (m_v - m_v.mean()) / (m_v.std() + 1e-6)
    return (p_z - m_z).pow(2).mean()


def map_consistency(field_mean: Tensor, sbp: Tensor, dbp: Tensor) -> Tensor:
    """Agreement between the field's time average and the clinical MAP rule of thumb.

    Normalised by 100 mmHg so this term arrives at the optimiser as a dimensionless
    quantity of order one, rather than as a number in the hundreds that would need a
    correspondingly tiny weight to stay balanced.
    """
    rule = map_from_sbp_dbp(sbp, dbp)
    return ((field_mean - rule) / 100.0).pow(2).mean()


def physiological_prior(params, sbp: Tensor, dbp: Tensor) -> Tensor:
    """One-sided hinges keeping parameters and outputs inside physiological ranges.

    Hinges, not squared penalties: inside the range the cost and the gradient are both
    exactly zero, so a subject who genuinely sits near the edge of normal is not dragged
    toward the population mean. Only genuinely impossible values are pushed on.

    The bounds are generous on purpose -- they are meant to exclude the impossible, not to
    encode what is typical. Hypertensive subjects are the clinically interesting ones and
    must not be regularised away.
    """

    def below(x: Tensor, lo: float) -> Tensor:
        return F.relu(lo - x).pow(2)

    def above(x: Tensor, hi: float) -> Tensor:
        return F.relu(x - hi).pow(2)

    def band(x: Tensor, lo: float, hi: float, scale: float) -> Tensor:
        return ((below(x, lo) + above(x, hi)) / (scale**2)).mean()

    cost = (
        band(params.R, 0.4, 2.5, 1.0)
        + band(params.C, 0.4, 4.0, 1.0)
        + band(params.SV, 25.0, 140.0, 50.0)
        + band(params.Tsys, 0.15, 0.45, 0.1)
        + band(params.HR, 35.0, 180.0, 50.0)
        + band(sbp, 60.0, 220.0, 50.0)
        + band(dbp, 30.0, 140.0, 50.0)
    )

    # Pulse pressure must be positive and physiologically bounded. A collapsed
    # PP is the classic failure mode when the input window is pure artifact: the
    # cheapest way to minimise data loss on garbage is to emit the cohort mean twice.
    pp = sbp - dbp
    cost = cost + (below(pp, 15.0) + above(pp, 110.0)).mean() / (25.0**2)

    return cost
