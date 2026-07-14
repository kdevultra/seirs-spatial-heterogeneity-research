import unittest

import numpy as np

from src.core.SEIRSParameters import SEIRSParameters
from src.core.utils import initial_conditions


class TestSeirsSanity(unittest.TestCase):
    def setUp(self):
        self.params = SEIRSParameters(
            dS=0.1,
            dE=0.1,
            dI=0.01,
            dR=0.1,
            N=1.0,
            grid_size=10,
            alpha=0.1,
            sigma=0.1,
            beta_0=0.21,
            beta_coeff=1.0,
            gamma_0=0.21,
            gamma_coeff=1.0,
        )

    def test_initial_conditions_mass_is_preserved(self):
        ic = initial_conditions(self.params, epsilon=0.1, center=(0.5, 0.5), radius=0.1)
        M = self.params.grid_size**2
        S0 = ic[:M].reshape(self.params.grid_size, self.params.grid_size)
        E0 = ic[M:2*M].reshape(self.params.grid_size, self.params.grid_size)
        I0 = ic[2*M:3*M].reshape(self.params.grid_size, self.params.grid_size)
        R0 = ic[3*M:].reshape(self.params.grid_size, self.params.grid_size)

        total_mass = np.trapezoid(np.trapezoid(S0 + E0 + I0 + R0, dx=self.params.h, axis=1), dx=self.params.h, axis=0)
        self.assertAlmostEqual(total_mass, self.params.N, places=6)

    def test_initial_conditions_are_non_negative(self):
        ic = initial_conditions(self.params, epsilon=0.1, center=(0.5, 0.5), radius=0.1)
        self.assertTrue(np.all(ic >= 0), "Initial condition contains negative values")

    def test_beta_gamma_field_properties(self):
        beta = self.params.beta_field(mu=(0.75, 0.75), sigma=0.1)
        gamma = self.params.gamma_field(mu=(0.25, 0.25), sigma=0.1)

        self.assertEqual(beta.shape, (self.params.grid_size, self.params.grid_size))
        self.assertEqual(gamma.shape, (self.params.grid_size, self.params.grid_size))
        self.assertTrue(np.all(beta > 0), "Beta field must be positive")
        self.assertTrue(np.all(gamma > 0), "Gamma field must be positive")
        self.assertGreater(beta.max(), beta.min(), "Beta field should vary across space")
        self.assertGreater(gamma.max(), gamma.min(), "Gamma field should vary across space")


if __name__ == '__main__':
    unittest.main()
