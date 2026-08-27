"""Training loop for the PINN and its baselines.

One trainer for all three model kinds so that the optimiser, schedule, early stopping and
data path are provably identical across them. When the results table says the PINN beat the
matched CNN, the only thing that differed is the model and its loss -- that is enforced
here, structurally, rather than promised in a README.

Two details that materially affect the numbers:

**Physics ramp.** The physics weights are scaled by ``min(1, epoch/physics_warmup_epochs)``.
At initialisation the pressure field is nearly flat, so its ODE residual is large and
carries almost no useful direction; applying full physics from step zero makes the field
collapse onto the smoothest waveform satisfying the ODE for some degenerate parameter set,
and the data term then has to climb back out. Letting the data term rough out a waveform
first and tightening physics afterwards is both faster and more stable. This is a real
hyperparameter, not a formality -- ``docs/RESULTS.md`` reports what happens without it.

**Early stopping on validation MAE, not on validation loss.** The two are different
objectives for the PINN: total loss includes physics terms that a model can reduce while
getting clinically worse. Selecting on the quantity that is actually reported avoids
choosing a checkpoint that is merely physically tidy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, save_config
from .data.build import build_dataset, training_mean
from .data.datasets import BPWindows, collate
from .metrics import regression_report
from .models.baselines import DataDrivenCNN, RidgeBaseline
from .models.pinn import PINNBP
from .physics.losses import (
    huber,
    map_consistency,
    ode_residual,
    periodicity,
    physiological_prior,
    tau_consistency,
)

__all__ = ["TrainResult", "train_model", "build_model", "predict"]


@dataclass
class TrainResult:
    """What a completed run leaves behind."""

    model: object
    history: list[dict]
    best_epoch: int
    best_val_mae: float
    test_report: dict
    train_mean: np.ndarray
    out_dir: Path
    seconds: float


def build_model(cfg: Config):
    """Instantiate the model named by ``cfg.model_kind``."""
    m = cfg.model
    if cfg.model_kind == "pinn":
        return PINNBP(
            in_channels=3,
            widths=tuple(m.widths),
            latent_dim=m.latent_dim,
            n_harmonics=m.n_harmonics,
            field_hidden=m.field_hidden,
            basis=m.basis,
            n_dense=m.n_dense,
            dropout=m.dropout,
        )
    if cfg.model_kind == "cnn":
        return DataDrivenCNN(
            in_channels=3,
            widths=tuple(m.widths),
            latent_dim=m.latent_dim,
            hidden=m.field_hidden,
            dropout=m.dropout,
        )
    if cfg.model_kind == "ridge":
        return RidgeBaseline()
    raise ValueError(f"unknown model_kind {cfg.model_kind!r} (expected pinn | cnn | ridge)")


def _pinn_losses(model: PINNBP, batch: dict, cfg: Config, ramp: float, gen: torch.Generator):
    """Forward pass plus every loss term, already weighted and summed.

    Returns:
        ``(total, parts_dict, out)``.
    """
    out = model(batch["x"])
    params = out.params
    t = cfg.train

    def field_fn(z, tt):
        return model.field(z, tt, params.period)

    data = huber(out.sbp, batch["sbp"]) + huber(out.dbp, batch["dbp"])
    ode = ode_residual(field_fn, out.latent, params, n_collocation=t.n_collocation, generator=gen)
    per = periodicity(field_fn, out.latent, params)
    tau = tau_consistency(params, batch["tau"], batch["tau_valid"])
    mapc = map_consistency(out.map, out.sbp, out.dbp)
    prior = physiological_prior(params, out.sbp, out.dbp)

    total = (
        t.w_data * data
        + ramp * (
            t.w_ode * ode
            + t.w_periodicity * per
            + t.w_tau * tau
            + t.w_map * mapc
            + t.w_prior * prior
        )
    )

    parts = {
        "data": float(data.detach()),
        "ode": float(ode.detach()),
        "per": float(per.detach()),
        "tau": float(tau.detach()),
        "map": float(mapc.detach()),
        "prior": float(prior.detach()),
    }
    return total, parts, out


@torch.no_grad()
def predict(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run a model over a loader and collect predictions and truth.

    Returns:
        ``(y_true, y_pred, extras)`` with ``(n, 2)`` arrays in (SBP, DBP) order. ``extras``
        carries the predicted physical parameters when the model has them, so parameter
        recovery can be checked without a second pass.
    """
    model.eval()
    yt: list[np.ndarray] = []
    yp: list[np.ndarray] = []
    par: list[np.ndarray] = []
    sqi: list[np.ndarray] = []

    for batch in loader:
        out = model(batch["x"])
        yp.append(np.stack([out.sbp.numpy(), out.dbp.numpy()], axis=1))
        yt.append(np.stack([batch["sbp"].numpy(), batch["dbp"].numpy()], axis=1))
        sqi.append(batch["sqi"].numpy())
        if hasattr(out, "params"):
            p = out.params
            par.append(
                np.stack([p.R.numpy(), p.C.numpy(), p.SV.numpy(), p.HR.numpy(), p.Tsys.numpy()],
                         axis=1)
            )

    extras = {"sqi": np.concatenate(sqi)}
    if par:
        extras["params"] = np.concatenate(par)
    return np.concatenate(yt), np.concatenate(yp), extras


def _fit_ridge(cfg: Config, prepared: dict, splits, out_dir: Path, t0: float) -> TrainResult:
    """Ridge path: no epochs, no torch, but the same splits and the same report."""
    tr = BPWindows(prepared, splits.train)
    te = BPWindows(prepared, splits.test)

    model = RidgeBaseline().fit(tr.features, tr.targets, groups=tr.subjects)
    y_pred = model.predict(te.features)

    tmean = training_mean(prepared, splits)
    report = regression_report(te.targets, y_pred, train_mean=tmean)
    report["best_alpha"] = model.best_alpha_

    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TrainResult(
        model=model, history=[], best_epoch=0,
        best_val_mae=float("nan"), test_report=report, train_mean=tmean,
        out_dir=out_dir, seconds=time.time() - t0,
    )


def train_model(cfg: Config, verbose: bool = True) -> TrainResult:
    """Train and evaluate one model end to end.

    Writes the config, the training history, the best checkpoint and the test report into
    ``<out_dir>/<name>/``. Everything needed to reproduce or audit the run ends up on disk,
    because a number in a terminal that has since scrolled away is not a result.
    """
    t0 = time.time()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    # Leave a thread free: on a 6-thread part, saturating every thread makes the machine
    # unresponsive and, with this little RAM, slower overall than using five.
    torch.set_num_threads(max(1, min(5, torch.get_num_threads())))

    out_dir = Path(cfg.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    prepared, splits = build_dataset(cfg, verbose=verbose)

    if cfg.model_kind == "ridge":
        return _fit_ridge(cfg, prepared, splits, out_dir, t0)

    train_ds = BPWindows(prepared, splits.train)
    val_ds = BPWindows(prepared, splits.val)
    test_ds = BPWindows(prepared, splits.test)

    dl = dict(batch_size=cfg.train.batch_size, collate_fn=collate,
              num_workers=cfg.train.num_workers)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)
    test_dl = DataLoader(test_ds, shuffle=False, **dl)

    model = build_model(cfg)
    n_par = model.n_parameters()
    if verbose:
        print(f"[model] {cfg.model_kind}  {n_par:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    steps_per_epoch = max(1, len(train_dl))
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        """Linear warmup then cosine decay, as a multiplier on the base lr."""
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator().manual_seed(cfg.train.seed)

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    since_improved = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        ramp = min(1.0, (epoch + 1) / max(1, cfg.train.physics_warmup_epochs))
        agg: dict[str, float] = {}
        n_batches = 0

        for batch in train_dl:
            opt.zero_grad(set_to_none=True)

            if cfg.model_kind == "pinn":
                loss, parts, _ = _pinn_losses(model, batch, cfg, ramp, gen)
            else:
                out = model(batch["x"])
                loss = huber(out.sbp, batch["sbp"]) + huber(out.dbp, batch["dbp"])
                parts = {"data": float(loss.detach())}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
            n_batches += 1

        y_true, y_pred, _ = predict(model, val_dl)
        val_mae_sbp = float(np.abs(y_pred[:, 0] - y_true[:, 0]).mean())
        val_mae_dbp = float(np.abs(y_pred[:, 1] - y_true[:, 1]).mean())
        val_mae = 0.5 * (val_mae_sbp + val_mae_dbp)

        row = {k: v / max(1, n_batches) for k, v in agg.items()}
        row.update({
            "epoch": epoch,
            "lr": float(sched.get_last_lr()[0]),
            "ramp": ramp,
            "val_mae_sbp": val_mae_sbp,
            "val_mae_dbp": val_mae_dbp,
            "val_mae": val_mae,
        })
        history.append(row)

        improved = val_mae < best_val - 1e-4
        if improved:
            best_val, best_epoch = val_mae, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if verbose:
            extra = ""
            if cfg.model_kind == "pinn":
                extra = (f" ode {row.get('ode', 0):.4f} tau {row.get('tau', 0):.4f}"
                         f" prior {row.get('prior', 0):.4f}")
            print(
                f"  epoch {epoch:3d}  loss {row['loss']:7.3f}  "
                f"val MAE {val_mae:5.2f} (S {val_mae_sbp:5.2f} / D {val_mae_dbp:5.2f})"
                f"{extra}{'  *' if improved else ''}"
            )

        if since_improved >= cfg.train.patience:
            if verbose:
                print(f"[train] early stop at epoch {epoch} (no gain for {since_improved})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "config": cfg.as_dict(), "epoch": best_epoch},
                   out_dir / "best.pt")

    y_true, y_pred, extras = predict(model, test_dl)
    tmean = training_mean(prepared, splits)
    report = regression_report(y_true, y_pred, train_mean=tmean)
    report["n_parameters"] = n_par
    report["best_epoch"] = best_epoch
    report["best_val_mae"] = best_val
    report["train_seconds"] = time.time() - t0

    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred,
                        **extras)

    if verbose:
        s, d = report["sbp"], report["dbp"]
        print(
            f"[test] SBP MAE {s['mae']:.2f}  ME {s['me']:+.2f}  SD {s['sd']:.2f}  "
            f"BHS {s['bhs']['grade']}  |  DBP MAE {d['mae']:.2f}  ME {d['me']:+.2f}  "
            f"SD {d['sd']:.2f}  BHS {d['bhs']['grade']}"
        )
        if "skill_vs_mean" in report:
            sk = report["skill_vs_mean"]
            print(f"[test] error reduction vs mean predictor: "
                  f"SBP {sk['sbp'] * 100:+.1f}%  DBP {sk['dbp'] * 100:+.1f}%")

    return TrainResult(
        model=model, history=history, best_epoch=best_epoch, best_val_mae=best_val,
        test_report=report, train_mean=tmean, out_dir=out_dir, seconds=time.time() - t0,
    )
