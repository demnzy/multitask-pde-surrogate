import time
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def simulate_heat(
    sources: torch.Tensor,
    grid_size: int = 32,
    steps: int = 100,
    source_radius: float = 0.08,
    return_timing: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
    """
    Finite-difference 2D steady-state heat diffusion simulation using Jacobi relaxation.

    Governing equation:
        \\nabla^2 T(x, y) = -Q(x, y)
    Subject to homogeneous Dirichlet boundary conditions (T=0 on boundaries).

    Parameters
    ----------
    sources : torch.Tensor
        Tensor of shape (N, 3) where each row is [x_center, y_center, power].
    grid_size : int, default=32
        Spatial grid resolution (grid_size x grid_size).
    steps : int, default=100
        Number of Jacobi relaxation iterations.
    source_radius : float, default=0.08
        Characteristic Gaussian heat source width (sigma).
    return_timing : bool, default=False
        If True, returns a tuple of (temperature_grid, elapsed_seconds).

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, float]
        2D temperature field of shape (grid_size, grid_size), optionally with execution time.
    """
    start_time = time.perf_counter()
    dx = 1.0 / (grid_size - 1)

    x = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    # Accumulate heat source generation term Q(x, y)
    Q = torch.zeros((1, 1, grid_size, grid_size), dtype=torch.float32)
    for src in sources:
        sx, sy, power = float(src[0]), float(src[1]), float(src[2])
        dist_sq = (grid_x - sx) ** 2 + (grid_y - sy) ** 2
        Q += power * torch.exp(-dist_sq / (2.0 * (source_radius ** 2)))

    # 5-point discrete Laplacian stencil for Jacobi relaxation
    stencil = torch.tensor(
        [
            [0.00, 0.25, 0.00],
            [0.25, 0.00, 0.25],
            [0.00, 0.25, 0.00],
        ],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)

    T = torch.zeros((1, 1, grid_size, grid_size), dtype=torch.float32)
    q_term = 0.25 * (dx ** 2) * Q

    # Jacobi iteration loop
    for _ in range(steps):
        T = F.conv2d(T, stencil, padding=1) + q_term
        # Enforce cold Dirichlet boundaries
        T[:, :, 0, :] = 0.0
        T[:, :, -1, :] = 0.0
        T[:, :, :, 0] = 0.0
        T[:, :, :, -1] = 0.0

    elapsed = time.perf_counter() - start_time
    grid_output = T.squeeze(0).squeeze(0)

    if return_timing:
        return grid_output, elapsed
    return grid_output


def get_peak_temperature(
    sources: torch.Tensor,
    fidelity: int = 0,
) -> Tuple[float, float]:
    """
    Evaluates the peak plate temperature and runtime at a specified fidelity level.

    Fidelity levels:
        fidelity=0 : Low-fidelity coarse grid (16x16, 40 steps, ~15-20 ms).
        fidelity=1 : High-fidelity fine grid (64x64, 400 steps, ~70-100 ms).

    Parameters
    ----------
    sources : torch.Tensor
        Tensor of shape (N, 3) representing heater sources.
    fidelity : int, default=0
        Fidelity index (0 = coarse / low-fidelity, 1 = fine / high-fidelity).

    Returns
    -------
    Tuple[float, float]
        (peak_temperature_celsius, execution_time_seconds)
    """
    if fidelity == 0:
        grid, runtime = simulate_heat(sources, grid_size=16, steps=40, return_timing=True)
    elif fidelity == 1:
        grid, runtime = simulate_heat(sources, grid_size=64, steps=400, return_timing=True)
    else:
        raise ValueError(f"Unsupported fidelity level: {fidelity}. Must be 0 or 1.")

    peak_temp = float(grid.max().item())
    return peak_temp, runtime


def generate_random_heaters(
    n_heaters: int = 2,
    rng: Optional[np.random.RandomState] = None,
) -> torch.Tensor:
    """
    Generates random spatial positions and thermal power outputs for heater elements.

    Parameters
    ----------
    n_heaters : int, default=2
        Number of heating elements.
    rng : Optional[np.random.RandomState], default=None
        Numpy random state for reproducibility.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_heaters, 3) where each row is [x, y, power].
    """
    gen = rng if rng is not None else np.random

    if n_heaters == 2:
        h1 = [gen.uniform(0.2, 0.4), gen.uniform(0.2, 0.8), gen.uniform(5.0, 15.0)]
        h2 = [gen.uniform(0.6, 0.8), gen.uniform(0.2, 0.8), gen.uniform(5.0, 15.0)]
        return torch.tensor([h1, h2], dtype=torch.float32)

    heaters = []
    for _ in range(n_heaters):
        h = [gen.uniform(0.15, 0.85), gen.uniform(0.15, 0.85), gen.uniform(5.0, 15.0)]
        heaters.append(h)
    return torch.tensor(heaters, dtype=torch.float32)
