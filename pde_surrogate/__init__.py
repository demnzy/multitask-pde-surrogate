"""
Multi-fidelity Gaussian Process surrogate modeling and cost-aware active learning
for 2D steady-state heat diffusion PDEs.
"""

from .solver import (
    simulate_heat,
    get_peak_temperature,
    generate_random_heaters,
)
from .model import (
    MultiFidelityGP,
    fit_gp_surrogate,
    predict_fidelity,
)
from .active_learning import (
    CostAwareAcquisition,
    run_active_learning,
)

__all__ = [
    "simulate_heat",
    "get_peak_temperature",
    "generate_random_heaters",
    "MultiFidelityGP",
    "fit_gp_surrogate",
    "predict_fidelity",
    "CostAwareAcquisition",
    "run_active_learning",
]
