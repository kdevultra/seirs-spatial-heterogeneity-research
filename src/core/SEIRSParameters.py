# TODO : To be reviewed

from dataclasses import dataclass
from numpy import cos, pi, sqrt
import numpy as np

@dataclass(frozen=True)
class SEIRSParameters:
    # diffusion rates
    dS: float
    dE: float
    dI: float
    dR: float

    N: float # population density

    # spatial discretization paramters
    grid_size: int # number of nodes per direction

    # epidemiological parameters
    alpha: float # constant
    sigma: float # constant

    # paramters for heterogeneous beta and gamma
    beta_0: float
    beta_coeff: float
    gamma_0: float
    gamma_coeff: float

    @property
    def h(self) -> float:
        """Mesh spacing."""
        return 1 / (self.grid_size - 1)
    
    def __post_init__(self):
        # Check that the parameters are valid
        assert self.grid_size > 1, "grid_size must be greater than 1"
        assert self.beta_0 > 0, "beta_0 must be positive"
        assert self.gamma_0 > 0, "gamma_0 must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert self.alpha > 0, "alpha must be positive"
        assert self.N > 0, "N must be positive"
        assert self.dS >= 0 and self.dE >= 0 and self.dI >= 0 and self.dR >= 0, "diffusion rates must be non-negative"

    def make_grid(self):
        """Return the grids X, Y on [0,1]×[0,1] with n nodes per direction."""
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y, indexing='ij')   # shape (n, n), indexing ij → X[i,j]=x[i]
        return X, Y

    # heterogeneity parameters for beta and gamma fields

    def gamma_field(self, mu, sigma, constant = None) -> np.ndarray:
        """Heterogeneous recovery rate. A 2D extension of the 1D function used by Song et al. (2019)"""
        if constant is None:
            constant = self.gamma_0
        return self.field_perturbated_with_gaussian(mu, sigma, constant, coeff=self.gamma_coeff)

    def beta_field(self, mu, sigma, constant = None) -> np.ndarray:
        """Heterogeneous transmission rate, as used by Yang et al. (2021)."""
        if constant is None:
            constant = self.beta_0
        return self.field_perturbated_with_gaussian(mu, sigma, constant, coeff=self.beta_coeff)
    
    def field_perturbated_with_gaussian(self, mu:tuple, sigma, constant, coeff=1.0):
        """
        mu: center in 2D
        sigma: identical in x and y
        """
        X, Y = self.make_grid()
        B = constant + coeff * ( ( 1/(sigma**2) )
                           * np.exp(-(1/(2*sigma**2)) * ((X - mu[0])**2 + (Y - mu[1])**2))
                        )
        return B


