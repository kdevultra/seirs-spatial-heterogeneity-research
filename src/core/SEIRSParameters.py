# TODO : To be reviewed

from dataclasses import dataclass
from numpy import cos, pi
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

    # paramters for heterogeneous beta
    beta_0: float
    c1: float
    c2: float

    @property
    def h(self) -> float:
        """Mesh spacing."""
        return 1 / (self.grid_size - 1)
    
    def __post_init__(self):
        # Check that the parameters are valid
        assert self.grid_size > 1, "grid_size must be greater than 1"
        assert self.beta_0 > 0, "beta_0 must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert 0 < self.c1 < 1 and 0 < self.c2 < 1, "c1 and c2 must be in (0,1)"
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

    def gamma_field(self) -> np.ndarray:
        """Heterogeneous recovery rate. A 2D extension of the 1D function used by Song et al. (2019)"""
        X, Y = self.make_grid()
        return X + Y + 1.0

    def beta_field(self) -> np.ndarray:
        """Heterogeneous transmission rate, as used by Yang et al. (2021)."""
        X, Y = self.make_grid()
        B = self.beta_0 * (1 + self.c1 * cos(pi * X)) * (1 + self.c2 * cos(pi * Y))
        return B

