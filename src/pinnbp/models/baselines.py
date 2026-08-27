"""Baselines the PINN has to beat to justify itself.

A physics-informed model is a more complicated thing than a regressor, and the extra
complexity has to buy something measurable. These two baselines are what make that claim
falsifiable:

``DataDrivenCNN``
    The **matched ablation**, and the important one. Identical encoder, identical width,
    identical depth, identical input channels, identical optimiser and schedule. The only
    differences are that it regresses SBP and DBP directly instead of through a pressure
    field, and that it has no physics terms in its loss. Any gap between it and the PINN is
    therefore attributable to the physics and to nothing else. It is *not* a weakened
    strawman: it has slightly more head capacity than the PINN's parameter head, so if it
    loses it does not lose on parameter count.

``RidgeBaseline``
    The **sanity floor**. Twenty handcrafted pulse-wave-analysis features and a linear
    model. This exists because a substantial part of the cuffless-BP literature reports deep
    models that a linear fit on classical features matches, and a project that does not run
    that check cannot know which side of that line it is on. If ridge matches the deep
    models here, that is the honest headline and it goes in ``docs/RESULTS.md`` as such.

There is a third, unwritten baseline that ``pinnbp.metrics`` computes automatically: always
predicting the training-set mean. On a between-subject task with a narrow pressure
distribution, mean prediction is a surprisingly strong competitor, and reporting MAE without
it next to it is close to meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .encoder import PPGEncoder

__all__ = ["DataDrivenCNN", "CNNOutput", "RidgeBaseline", "MeanPredictor"]


@dataclass
class CNNOutput:
    """Mirrors the fields of :class:`~pinnbp.models.pinn.PINNOutput` that evaluation uses.

    Deliberately the same attribute names so that ``evaluate`` and the plotting code are
    model-agnostic and there is no per-model branch that could accidentally treat the two
    differently.
    """

    sbp: Tensor
    dbp: Tensor
    latent: Tensor


class DataDrivenCNN(nn.Module):
    """Encoder plus a direct SBP/DBP regression head.

    The output is parameterised as ``(centre + scale * raw)`` with centre 120/75 mmHg,
    matching the PINN's field centring. Without it this baseline would spend its first
    epochs travelling from zero to the physiological range while the PINN starts there --
    a difference in convergence speed that has nothing to do with physics and would
    contaminate the comparison at any fixed epoch budget.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 48, 64, 96),
        latent_dim: int = 128,
        hidden: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.encoder = PPGEncoder(in_channels, widths, latent_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

        self.register_buffer("centre", torch.tensor([120.0, 75.0]))
        self.register_buffer("scale", torch.tensor([25.0, 15.0]))

    def forward(self, x: Tensor) -> CNNOutput:
        z = self.encoder(x)
        out = self.centre + self.scale * self.head(z)
        return CNNOutput(sbp=out[:, 0], dbp=out[:, 1], latent=z)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RidgeBaseline:
    """Ridge regression on classical pulse-wave-analysis features.

    Median imputation before scaling, because feature extraction legitimately returns NaN
    when a window has no measurable dicrotic notch or too few beats to fit -- and on heavily
    corrupted windows that is most of them. Dropping those rows would quietly restrict the
    baseline to the easy examples and flatter it exactly where the robustness comparison
    matters most.

    Alpha is chosen by cross-validation *grouped by subject*, matching the main split. Plain
    k-fold here would leak subjects into the validation folds and select an alpha tuned to a
    leaky estimate.
    """

    def __init__(self, alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.alphas = alphas
        self.pipeline = None
        self.best_alpha_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> RidgeBaseline:
        """Fit on ``(n, F)`` features and ``(n, 2)`` SBP/DBP targets."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

        def make(alpha: float) -> Pipeline:
            return Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )

        best_alpha, best_score = self.alphas[0], -np.inf
        if groups is not None and len(np.unique(groups)) >= 3:
            n_splits = int(min(5, len(np.unique(groups))))
            cv = GroupKFold(n_splits=n_splits)
            for a in self.alphas:
                score = float(
                    np.mean(cross_val_score(make(a), X, y, groups=groups, cv=cv,
                                            scoring="neg_mean_absolute_error"))
                )
                if score > best_score:
                    best_alpha, best_score = a, score

        self.best_alpha_ = best_alpha
        self.pipeline = make(best_alpha).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ``(n, 2)`` SBP/DBP."""
        if self.pipeline is None:
            raise RuntimeError("RidgeBaseline.predict called before fit")
        X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return np.asarray(self.pipeline.predict(X))


class MeanPredictor:
    """Predict the training-set mean for every window.

    The floor every other model must clear. On a between-subject cuffless-BP task this is
    much harder to beat than it sounds, and a reported MAE that does not beat it comfortably
    means the model has learned the cohort's average pressure and little else.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> MeanPredictor:
        self.mean_ = np.asarray(y, dtype=np.float64).mean(axis=0)
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("MeanPredictor.predict called before fit")
        return np.tile(self.mean_, (n, 1))
