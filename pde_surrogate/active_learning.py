from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import gpytorch

from .solver import get_peak_temperature, generate_random_heaters
from .model import MultiFidelityGP, fit_gp_surrogate


class CostAwareAcquisition:
    """
    Cost-aware uncertainty acquisition function for multi-fidelity surrogate active learning.

    Acquisition criterion:
        alpha(x, m) = sigma_m^2(x) / c_m

    where:
        sigma_m^2(x) is the posterior predictive variance at fidelity m.
        c_m is the computational cost (execution time) of fidelity m.
    """

    def __init__(self, cost_low: float = 17.0, cost_high: float = 70.0):
        self.cost_low = max(cost_low, 1e-6)
        self.cost_high = max(cost_high, 1e-6)

    def evaluate_candidates(
        self,
        model: MultiFidelityGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes acquisition scores for all candidates at low and high fidelity.

        Parameters
        ----------
        model : MultiFidelityGP
            Trained multi-fidelity model.
        likelihood : GaussianLikelihood
            Trained likelihood.
        candidates : torch.Tensor
            Candidate design features of shape (N, D).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (scores_low_fidelity, scores_high_fidelity) of shapes (N,).
        """
        n_cand = candidates.shape[0]
        c_low = torch.cat([candidates, torch.zeros(n_cand, 1, dtype=torch.float32)], dim=-1)
        c_high = torch.cat([candidates, torch.ones(n_cand, 1, dtype=torch.float32)], dim=-1)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            var_low = likelihood(model(c_low)).variance
            var_high = likelihood(model(c_high)).variance

        score_low = var_low / self.cost_low
        score_high = var_high / self.cost_high
        return score_low, score_high

    def select_best_query(
        self,
        model: MultiFidelityGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        candidates: torch.Tensor,
    ) -> Tuple[int, int, float]:
        """
        Finds candidate and fidelity level that maximize the acquisition score.

        Returns
        -------
        Tuple[int, int, float]
            (best_candidate_index, best_fidelity_level, max_acquisition_score)
        """
        score_low, score_high = self.evaluate_candidates(model, likelihood, candidates)
        max_low, idx_low = score_low.max(dim=0)
        max_high, idx_high = score_high.max(dim=0)

        if max_low.item() >= max_high.item():
            return int(idx_low.item()), 0, float(max_low.item())
        return int(idx_high.item()), 1, float(max_high.item())


def run_active_learning(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int = 5,
    pool_size: int = 20,
    cost_low: float = 17.0,
    cost_high: float = 70.0,
    eval_pool_size: int = 20,
    retrain_epochs: int = 40,
    lr: float = 0.08,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = True,
) -> Tuple[MultiFidelityGP, gpytorch.likelihoods.GaussianLikelihood, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]:
    """
    Executes cost-aware multi-fidelity active learning loop.

    Parameters
    ----------
    train_x : torch.Tensor
        Initial training inputs of shape (N, 7).
    train_y : torch.Tensor
        Initial targets of shape (N,).
    n_iterations : int, default=5
        Number of active learning query iterations.
    pool_size : int, default=20
        Number of candidate designs proposed at each iteration.
    cost_low : float, default=17.0
        Cost weight for low-fidelity queries.
    cost_high : float, default=70.0
        Cost weight for high-fidelity queries.
    eval_pool_size : int, default=20
        Size of holdout test pool to track variance reduction.
    retrain_epochs : int, default=40
        Optimization epochs per active query step.
    lr : float, default=0.08
        Adam learning rate.
    rng : Optional[np.random.RandomState], default=None
        Random state.
    verbose : bool, default=True
        Whether to print per-iteration query decisions.

    Returns
    -------
    Tuple[MultiFidelityGP, GaussianLikelihood, torch.Tensor, torch.Tensor, Dict[str, List]]
        Updated model, likelihood, final train_x, final train_y, and convergence history.
    """
    acquisition = CostAwareAcquisition(cost_low=cost_low, cost_high=cost_high)

    # Fixed evaluation pool for monitoring predictive variance reduction
    eval_pool = torch.stack([generate_random_heaters(2, rng=rng).flatten() for _ in range(eval_pool_size)])
    eval_high = torch.cat([eval_pool, torch.ones(eval_pool_size, 1, dtype=torch.float32)], dim=-1)

    # Initial model fit
    model, likelihood, _ = fit_gp_surrogate(train_x, train_y, num_epochs=retrain_epochs, lr=lr, verbose=False)

    with torch.no_grad():
        init_var = float(likelihood(model(eval_high)).variance.mean().item())

    history: Dict[str, List[Any]] = {
        "iteration": [0],
        "cumulative_time_ms": [0.0],
        "selected_fidelity": [None],
        "holdout_variance": [init_var],
        "peak_temperatures": [None],
    }

    total_time_spent = 0.0
    current_x = train_x.clone()
    current_y = train_y.clone()

    for loop_iter in range(1, n_iterations + 1):
        candidates = torch.stack([generate_random_heaters(2, rng=rng).flatten() for _ in range(pool_size)])
        cand_idx, chosen_fidelity, score = acquisition.select_best_query(model, likelihood, candidates)
        chosen_setup = candidates[cand_idx]

        # Simulate selected design
        sources = chosen_setup.view(2, 3)
        peak_temp, sim_runtime = get_peak_temperature(sources, fidelity=chosen_fidelity)
        total_time_spent += sim_runtime

        fid_str = "LOW (m=0)" if chosen_fidelity == 0 else "HIGH (m=1)"
        if verbose:
            print(f"Iter {loop_iter:02d}: Selected {fid_str} | Score: {score:.5f} | "
                  f"Runtime: {sim_runtime * 1000:.1f} ms | Peak: {peak_temp:.4f} °C")

        # Augment training set
        new_sample_x = torch.cat([chosen_setup, torch.tensor([float(chosen_fidelity)], dtype=torch.float32)]).unsqueeze(0)
        current_x = torch.cat([current_x, new_sample_x], dim=0)
        current_y = torch.cat([current_y, torch.tensor([peak_temp], dtype=torch.float32)], dim=0)

        # Retrain GP surrogate
        model, likelihood, _ = fit_gp_surrogate(current_x, current_y, num_epochs=retrain_epochs, lr=lr, verbose=False)

        # Measure variance reduction on holdout fine-grid configurations
        with torch.no_grad():
            cur_var = float(likelihood(model(eval_high)).variance.mean().item())

        history["iteration"].append(loop_iter)
        history["cumulative_time_ms"].append(total_time_spent * 1000.0)
        history["selected_fidelity"].append("Low" if chosen_fidelity == 0 else "High")
        history["holdout_variance"].append(cur_var)
        history["peak_temperatures"].append(peak_temp)

    return model, likelihood, current_x, current_y, history
