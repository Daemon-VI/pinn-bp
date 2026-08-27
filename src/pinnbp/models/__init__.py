"""Model definitions: the PINN and the baselines it is measured against."""

from .baselines import DataDrivenCNN, MeanPredictor, RidgeBaseline  # noqa: F401
from .pinn import PINNBP  # noqa: F401
