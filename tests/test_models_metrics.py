"""Tests for the models and the clinical metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinnbp.metrics import aami_check, bhs_grade, error_stats, regression_report
from pinnbp.models.baselines import DataDrivenCNN, MeanPredictor, RidgeBaseline
from pinnbp.models.pinn import PARAM_INIT, PARAM_RANGES, PINNBP
from pinnbp.physics.losses import ode_residual, periodicity, physiological_prior, tau_consistency


def batch(n: int = 8, length: int = 1000) -> torch.Tensor:
    return torch.randn(n, 3, length, generator=torch.Generator().manual_seed(0))


# --------------------------------------------------------------------------- models


def test_pinn_and_cnn_have_comparable_capacity():
    """The comparison is only fair if the ablation is not starved or inflated."""
    p, c = PINNBP().n_parameters(), DataDrivenCNN().n_parameters()
    assert 0.8 < c / p < 1.2, f"PINN {p}, CNN {c}"


def test_pinn_outputs_are_physiological_and_ordered():
    out = PINNBP()(batch())
    assert torch.all(out.sbp > out.dbp)
    assert torch.all((out.sbp > 40) & (out.sbp < 260))
    assert torch.all((out.dbp > 20) & (out.dbp < 180))
    # MAP must lie strictly between them, since it is the mean of the same waveform.
    assert torch.all((out.map > out.dbp) & (out.map < out.sbp))


def test_predicted_parameters_are_hard_bounded():
    """Bounds are structural, not penalised, so they must hold for any input."""
    out = PINNBP()(torch.randn(16, 3, 1000) * 50)
    p = out.params
    for name, tensor in (("R", p.R), ("C", p.C), ("SV", p.SV), ("HR", p.HR), ("Tsys", p.Tsys)):
        lo, hi = PARAM_RANGES[name]
        assert torch.all((tensor >= lo) & (tensor <= hi)), name


def test_untrained_model_starts_at_a_healthy_adult():
    """Not the range midpoint -- that would put every subject at 112 bpm."""
    out = PINNBP()(batch(32))
    for name, tensor in (("R", out.params.R), ("C", out.params.C), ("HR", out.params.HR)):
        assert tensor.mean().item() == pytest.approx(PARAM_INIT[name], rel=0.05), name


def test_harmonic_basis_is_periodic_by_construction():
    """The hard constraint must hold to machine precision; the raw basis must not."""
    m = PINNBP(basis="harmonic")
    out = m(batch())
    ff = lambda z, t: m.field(z, t, out.params.period)  # noqa: E731
    assert periodicity(ff, out.latent, out.params).item() < 1e-8

    m2 = PINNBP(basis="raw")
    o2 = m2(batch())
    ff2 = lambda z, t: m2.field(z, t, o2.params.period)  # noqa: E731
    assert periodicity(ff2, o2.latent, o2.params).item() > 1e-8


def test_ode_residual_is_finite_and_differentiable():
    m = PINNBP()
    out = m(batch())
    ff = lambda z, t: m.field(z, t, out.params.period)  # noqa: E731
    r = ode_residual(ff, out.latent, out.params, n_collocation=32)
    assert torch.isfinite(r) and r.item() >= 0
    r.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_prior_penalises_a_collapsed_pulse_pressure():
    """The classic failure on pure artifact: emit the cohort mean twice."""
    m = PINNBP()
    out = m(batch())
    healthy = physiological_prior(out.params, torch.full((8,), 120.0), torch.full((8,), 78.0))
    collapsed = physiological_prior(out.params, torch.full((8,), 99.0), torch.full((8,), 98.0))
    assert collapsed.item() > healthy.item()


def test_tau_loss_is_scale_free():
    """Scaling the measured tau must not change the loss -- it constrains rank only."""
    m = PINNBP()
    out = m(batch(32))
    tau = torch.rand(32) * 2 + 0.5
    valid = torch.ones(32, dtype=torch.bool)
    a = tau_consistency(out.params, tau, valid)
    b = tau_consistency(out.params, tau * 4.0, valid)
    assert a.item() == pytest.approx(b.item(), rel=1e-4)


def test_tau_loss_returns_zero_when_too_few_valid():
    m = PINNBP()
    out = m(batch(8))
    valid = torch.zeros(8, dtype=torch.bool)
    assert tau_consistency(out.params, torch.ones(8), valid).item() == 0.0


def test_cnn_returns_the_same_field_names_as_the_pinn():
    """Evaluation is model-agnostic; a mismatch here would need a per-model branch."""
    out = DataDrivenCNN()(batch())
    assert hasattr(out, "sbp") and hasattr(out, "dbp") and hasattr(out, "latent")


# --------------------------------------------------------------------------- metrics


def test_error_stats_on_a_known_case():
    st = error_stats(np.array([100.0, 110.0, 120.0]), np.array([105.0, 105.0, 120.0]))
    assert st.mae == pytest.approx(10 / 3)
    assert st.me == pytest.approx(0.0)
    assert st.max_ae == pytest.approx(5.0)


def test_aami_uses_signed_error_sd_not_absolute():
    """A model can post a fine MAE and still fail AAMI; the two must not be conflated."""
    rng = np.random.default_rng(0)
    truth = rng.normal(120, 15, 200)
    biased = truth + 9.0  # tight but badly biased
    assert not aami_check(truth, biased).passes

    good = truth + rng.normal(0, 3, 200)
    assert aami_check(truth, good).passes


def test_aami_fails_below_the_minimum_sample_size():
    truth = np.full(10, 120.0)
    assert not aami_check(truth, truth).passes


def test_aami_result_carries_its_own_caveat():
    """The note must travel with the number, so it cannot be quoted as a device validation."""
    r = aami_check(np.full(100, 120.0), np.full(100, 120.0))
    assert "not a device validation" in r.note


def test_bhs_grades_move_the_right_way():
    truth = np.full(1000, 120.0)
    assert bhs_grade(truth, truth).grade == "A"
    rng = np.random.default_rng(0)
    assert bhs_grade(truth, truth + rng.normal(0, 25, 1000)).grade == "D"


def test_r2_is_zero_for_the_mean_predictor_and_negative_for_worse():
    rng = np.random.default_rng(1)
    truth = rng.normal(120, 15, 300)
    rep = regression_report(
        np.stack([truth, truth - 40], 1),
        np.stack([np.full(300, truth.mean()), np.full(300, truth.mean() - 40)], 1),
    )
    assert rep["sbp"]["r2"] == pytest.approx(0.0, abs=0.02)


def test_report_includes_the_mean_baseline_and_skill():
    rng = np.random.default_rng(2)
    truth = np.stack([rng.normal(120, 15, 200), rng.normal(78, 10, 200)], 1)
    pred = truth + rng.normal(0, 4, truth.shape)
    rep = regression_report(truth, pred, train_mean=truth.mean(axis=0))
    assert "mean_baseline" in rep and "skill_vs_mean" in rep
    # A model this good must beat predicting the mean.
    assert rep["skill_vs_mean"]["sbp"] > 0.5


def test_mean_predictor_predicts_the_training_mean():
    y = np.array([[120.0, 80.0], [140.0, 90.0]])
    m = MeanPredictor().fit(y)
    assert np.allclose(m.predict(3), np.tile([130.0, 85.0], (3, 1)))


def test_ridge_handles_nan_features():
    """Feature extraction legitimately returns NaN; dropping those rows would flatter it."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    X[::5, 3] = np.nan
    y = np.stack([X[:, 0] * 10 + 120, X[:, 1] * 5 + 78], 1)
    groups = np.repeat(np.arange(12), 5)
    pred = RidgeBaseline().fit(X, y, groups=groups).predict(X)
    assert pred.shape == (60, 2) and np.all(np.isfinite(pred))
