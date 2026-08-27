"""Experiment configuration.

One dataclass tree, loadable from YAML, and every run writes the exact config it used next
to its results. That is not bureaucracy: the robustness sweep produces a dozen numbers that
are only interpretable if the artifact severity, the split seed and the loss weights that
produced them are recoverable months later, and this project's own conventions require that
results be reproducible rather than asserted.

Defaults are tuned for this machine -- a 6-thread CPU with no GPU and roughly a gigabyte of
free memory. ``batch_size=64`` and ``epochs=60`` on the default synthetic cohort is about
10-14 minutes of wall clock. Raising ``n_subjects`` or ``n_dense`` moves that quickly; see
``docs/ARCHITECTURE.md`` for the measured costs before scaling anything up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "DataConfig", "ModelConfig", "TrainConfig", "load_config", "save_config"]


@dataclass
class DataConfig:
    """Where the data comes from and how it is cut up."""

    dataset: str = "synthetic"       # synthetic | ppgbp | uci
    root: str | None = None          # required for the real datasets
    n_subjects: int = 200
    duration_s: float = 60.0
    fs: float = 125.0
    window_s: float = 8.0
    stride_s: float = 4.0
    hypertensive_fraction: float = 0.35
    train_artifact_severity: float = 0.25
    train_artifact_prob: float = 0.5
    split: str = "subject"           # subject | random (random is the leakage demo only)
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 0
    cache_dir: str = "data/cache"


@dataclass
class ModelConfig:
    """Architecture. Shared by the PINN and the matched CNN baseline."""

    widths: tuple[int, ...] = (32, 48, 64, 96)
    latent_dim: int = 128
    n_harmonics: int = 8
    field_hidden: int = 128
    basis: str = "harmonic"          # harmonic | raw
    n_dense: int = 128
    dropout: float = 0.10


@dataclass
class TrainConfig:
    """Optimisation and loss weighting."""

    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    n_collocation: int = 64
    patience: int = 15
    seed: int = 0
    num_workers: int = 0             # >0 costs a process per worker; not worth it here
    # Loss weights; see pinnbp.physics.losses.LossWeights for what each term does.
    w_data: float = 1.0
    w_ode: float = 0.10
    w_periodicity: float = 0.05
    w_tau: float = 0.05
    w_map: float = 0.02
    w_prior: float = 0.10
    # Physics is ramped in over this many epochs rather than applied from step zero. An
    # untrained field is nearly flat, so its ODE residual is large and almost pure noise;
    # letting the data term establish a roughly correct waveform first, then tightening the
    # physics, converges faster and more reliably than applying both at full strength.
    physics_warmup_epochs: int = 10


@dataclass
class Config:
    """Top-level experiment configuration."""

    name: str = "pinn-default"
    model_kind: str = "pinn"         # pinn | cnn | ridge
    out_dir: str = "reports"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(cls, data: dict[str, Any]):
    """Build a dataclass from a dict, recursing into nested dataclass fields.

    Unknown keys raise rather than being ignored. A silently dropped ``w_ode`` in a config
    file would produce a run that looks like a physics ablation but is not one, and the
    resulting number would be wrong in a way nothing downstream could detect.
    """
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}

    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")

    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _from_dict(f.type, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value

    # Nested dataclasses arrive as dicts but their annotations are strings under
    # ``from __future__ import annotations``, so the is_dataclass check above misses them.
    for name, sub in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig)):
        if name in kwargs and isinstance(kwargs[name], dict):
            kwargs[name] = _from_dict(sub, kwargs[name])

    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the config beside a run's results, so the run stays reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.as_dict(), fh, sort_keys=False, default_flow_style=False)
