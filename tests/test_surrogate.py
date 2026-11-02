"""
Unit and integration tests for Multi-Fidelity Heat Diffusion PDE Surrogate.
"""

import unittest
import torch
from pde_surrogate import (
    simulate_heat,
    get_peak_temperature,
    generate_random_heaters,
    fit_gp_surrogate,
    predict_fidelity,
    CostAwareAcquisition,
)


class TestPDESurrogate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.sources = torch.tensor([[0.3, 0.4, 10.0], [0.7, 0.7, 8.0]], dtype=torch.float32)

    def test_simulate_heat_shape_and_boundaries(self):
        grid = simulate_heat(self.sources, grid_size=16, steps=10)
        self.assertEqual(grid.shape, (16, 16))
        # Verify cold Dirichlet boundary condition (T=0)
        self.assertTrue(torch.allclose(grid[0, :], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(grid[-1, :], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(grid[:, 0], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(grid[:, -1], torch.tensor(0.0)))
        # Inner temperature should be positive due to heat injection
        self.assertGreater(grid.max().item(), 0.0)

    def test_get_peak_temperature(self):
        peak_low, time_low = get_peak_temperature(self.sources, fidelity=0)
        peak_high, time_high = get_peak_temperature(self.sources, fidelity=1)

        self.assertGreater(peak_low, 0.0)
        self.assertGreater(peak_high, 0.0)
        self.assertGreater(time_low, 0.0)
        self.assertGreater(time_high, 0.0)
        # High fidelity simulation has 16x more spatial cells and 10x more iterations
        self.assertGreater(time_high, time_low)

    def test_generate_random_heaters(self):
        heaters = generate_random_heaters(n_heaters=2)
        self.assertEqual(heaters.shape, (2, 3))
        # Ensure values lie within expected plate domain [0, 1]
        self.assertTrue(torch.all(heaters[:, :2] >= 0.0))
        self.assertTrue(torch.all(heaters[:, :2] <= 1.0))
        # Powers should be positive
        self.assertTrue(torch.all(heaters[:, 2] > 0.0))

    def test_gp_training_and_inference(self):
        # Create minimal synthetic multi-fidelity dataset
        h1 = generate_random_heaters(2).flatten()
        h2 = generate_random_heaters(2).flatten()
        h3 = generate_random_heaters(2).flatten()

        train_x = torch.stack([
            torch.cat([h1, torch.tensor([0.0])]),
            torch.cat([h2, torch.tensor([0.0])]),
            torch.cat([h3, torch.tensor([1.0])]),
        ])
        train_y = torch.tensor([2.5, 3.1, 4.2], dtype=torch.float32)

        model, likelihood, losses = fit_gp_surrogate(train_x, train_y, num_epochs=10, lr=0.1)
        self.assertEqual(len(losses), 10)

        # Inference test
        test_h = generate_random_heaters(2).flatten()
        mean_hf, std_hf = predict_fidelity(model, likelihood, test_h, fidelity=1)
        self.assertEqual(mean_hf.shape, (1,))
        self.assertGreater(std_hf.item(), 0.0)

    def test_cost_aware_acquisition(self):
        acq = CostAwareAcquisition(cost_low=10.0, cost_high=100.0)
        self.assertEqual(acq.cost_low, 10.0)
        self.assertEqual(acq.cost_high, 100.0)


if __name__ == "__main__":
    unittest.main()
