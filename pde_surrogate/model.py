from typing import List, Optional, Tuple
import gpytorch
import torch


class MultiFidelityGP(gpytorch.models.ExactGP):
    """
    Multi-Fidelity Gaussian Process surrogate model using a separable Kronecker product
    between a spatial Matérn-5/2 kernel and a discrete inter-task IndexKernel.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs of shape (N, D + 1), where the final column is the discrete fidelity index.
    train_y : torch.Tensor
        Target values of shape (N,).
    likelihood : gpytorch.likelihoods.Likelihood
        Gaussian likelihood module.
    spatial_dim : int, default=6
        Number of physical design input features (excluding the fidelity index).
    num_tasks : int, default=2
        Number of discrete fidelity tasks (e.g. 2 for low and high).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        spatial_dim: int = 6,
        num_tasks: int = 2,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Spatial continuous kernel over design features
        self.spatial_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            active_dims=tuple(range(spatial_dim)),
        )

        # Discrete inter-task correlation kernel on the fidelity index
        self.task_kernel = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks,
            rank=1,
            active_dims=(spatial_dim,),
        )

        # Separable product covariance
        self.covar_module = self.spatial_kernel * self.task_kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp_surrogate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_epochs: int = 80,
    lr: float = 0.1,
    verbose: bool = False,
) -> Tuple[MultiFidelityGP, gpytorch.likelihoods.GaussianLikelihood, List[float]]:
    """
    Instantiates and optimizes a MultiFidelityGP using Adam and Exact MLL.

    Parameters
    ----------
    train_x : torch.Tensor
        Training features of shape (N, 7).
    train_y : torch.Tensor
        Target scalar values of shape (N,).
    num_epochs : int, default=80
        Optimization steps.
    lr : float, default=0.1
        Adam learning rate.
    verbose : bool, default=False
        Whether to print loss progress during training.

    Returns
    -------
    Tuple[MultiFidelityGP, GaussianLikelihood, List[float]]
        Fitted model, likelihood, and negative MLL loss history.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultiFidelityGP(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_history = []
    for step in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if verbose and (step + 1) % 20 == 0:
            print(f"Step {step+1:02d}/{num_epochs:02d} | Loss: {loss_val:.4f}")

    model.eval()
    likelihood.eval()
    return model, likelihood, loss_history


def predict_fidelity(
    model: MultiFidelityGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_designs: torch.Tensor,
    fidelity: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts posterior mean and standard deviation for candidate designs at a target fidelity.

    Parameters
    ----------
    model : MultiFidelityGP
        Trained GP model.
    likelihood : GaussianLikelihood
        Trained likelihood.
    x_designs : torch.Tensor
        Design features of shape (N, D) or (D,).
    fidelity : int, default=1
        Fidelity index to evaluate (0 for coarse, 1 for fine).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (predictive_means, predictive_stddevs)
    """
    if x_designs.dim() == 1:
        x_designs = x_designs.unsqueeze(0)

    n_samples = x_designs.shape[0]
    fid_col = torch.full((n_samples, 1), float(fidelity), dtype=torch.float32)
    inputs = torch.cat([x_designs, fid_col], dim=-1)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        dist = likelihood(model(inputs))
        means = dist.mean
        stddevs = dist.stddev

    return means, stddevs
