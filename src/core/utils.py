# laplacian, total mass check, 

import numpy as np
import scipy.sparse as sp

def build_1d_laplacian_neumann(n, h):
    """
    Second-order 1D Laplacian with Neumann boundary conditions (zero flux),
    constructed using ghost points.

    For an interior node i : (u[i-1] - 2u[i] + u[i+1]) / h²
    For node 0 (left boundary) : u[-1] = u[1]  →  (2u[1] - 2u[0]) / h²
        → main_diag[0] = -2/h²,  upper_diag[0] = +2/h²
    For node N-1 (right boundary) : u[N] = u[N-2] → (2u[N-2] - 2u[N-1]) / h²
        → main_diag[-1] = -2/h², lower_diag[-1] = +2/h²

    Parameters
    ----------
    n : int   — number of nodes
    h : float — mesh spacing

    Returns
    -------
    Sparse CSR matrix of size n×n.
    """
    
    main_diag   = -2.0 * np.ones(n)
    off_diag_lo =  1.0 * np.ones(n - 1)   # sous-diagonale (positions -1)
    off_diag_hi =  1.0 * np.ones(n - 1)   # sur-diagonale  (positions +1)

    # Neumann condition at node 0 (zero flux):
    #   row 0, coefficient at column 1  →  upper diagonal, index 0
    off_diag_hi[0] = 2.0

    # Neumann condition at node N-1 (zero flux):
    #   row N-1, coefficient at column N-2  →  lower diagonal, index N-2
    off_diag_lo[-1] = 2.0

    L = sp.diags(
        [off_diag_lo, main_diag, off_diag_hi],
        [-1, 0, 1], #type: ignore
        format='csr'
    )

    return L / h**2

def build_2d_laplacian_neumann(nx, ny, h):
    """
    2D Laplacian with Neumann boundary conditions, constructed via Kronecker products.

    Row-major indexing: node (i, j) → index k = i*ny + j.

    L2D = kron(I_y, L_x) + kron(L_y, I_x)

    Returned matrix size: (nx*ny) × (nx*ny).

    Parameters
    ----------
    nx, ny : int   — number of nodes in x and y
    h      : float — mesh spacing (assumed uniform hx = hy = h)
    """
    Lx = build_1d_laplacian_neumann(nx, h)
    Ly = build_1d_laplacian_neumann(ny, h)
    Ix = sp.eye(nx, format='csr')
    Iy = sp.eye(ny, format='csr')

    # With k = i*ny + j:
    #   x-derivatives  →  kron(Iy, Lx)
    #   y-derivatives  →  kron(Ly, Ix)
    return sp.kron(Iy, Lx) + sp.kron(Ly, Ix)

def mass(u, h):
    """
    Check the total mass of a 2D field u on a uniform grid with spacing h.
    Uses trapezoidal 2D integral.
    
    Parameters
    ----------
    u : 2D array of shape (nx, ny)
    h : float — mesh spacing

    Returns
    -------
    Total mass (float).
    """
    return np.trapezoid(np.trapezoid(u, dx=h, axis=1), dx=h)

def flatten(field):
    """Safe flattening of a 2D field into a 1D vector, preserving the row-major order."""
    return np.asarray(field).flatten(order='C')

def make_grid(grid_size):
    """Return the grids X, Y on [0,1]×[0,1] with n nodes per direction."""
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')   # shape (n, n), indexing ij → X[i,j]=x[i]
    return X, Y

def initial_conditions(params, epsilon=0.1, center=(0.1, 0.1), radius=0.08):
    """
    Build IC vectors satisfying ∫∫(S0+E0+I0+R0)dΩ = N exactly
    under the 2D trapezoidal rule.
    epsilon : fraction of N seeded as initial infected mass
    """
    X, Y = params.make_grid()
    h = params.h

    # Step 1: localized Gaussian blob centered near the high-risk corner
    dist_sq = (X - center[0])**2 + (Y - center[1])**2
    I_patch = np.exp(-dist_sq / (2 * radius**2))
    I_patch /= mass(I_patch, h)  # use the same quadrature as everything else

    # Step 2: scale to the requested total infected mass
    I0 = epsilon * params.N * I_patch
    E0 = 0.3 * I0
    Rec0 = np.zeros_like(I0)  # recovered compartment; renamed, R0 means reproduction number here

    # Step 3: uniform S0 so that total mass = N exactly
    ones_integral = mass(np.ones_like(I0), h)
    target_S_mass = params.N - mass(E0 + I0 + Rec0, h)
    S0 = (target_S_mass / ones_integral) * np.ones_like(I0)

    # Check what actually matters: I0's mass hit the target independent of
    # the algebraic S0 solve below, which is exact by construction and can't fail.
    I0_mass_err = abs(mass(I0, h) - epsilon * params.N)
    rel_tol = 1e-6
    assert I0_mass_err < rel_tol * max(params.N, 1.0), \
        f"I0 mass error = {I0_mass_err:.2e} " \
        f"(relative: {I0_mass_err/params.N:.2e}, should be < {rel_tol:.0e}); " \
        f"check I_patch normalization matches mass()'s quadrature"

    total_mass = mass(S0 + E0 + I0 + Rec0, h)
    assert abs(total_mass - params.N) < rel_tol * max(params.N, 1.0), \
        f"IC mass error = {abs(total_mass - params.N):.2e} " \
        f"(relative: {abs(total_mass - params.N)/params.N:.2e}, should be < {rel_tol:.0e})"

    assert (S0 > 0).all(), "S0 uniform background went negative — epsilon too large?"
    assert (I0 >= 0).all() and (E0 >= 0).all(), "Positivity violated in I0/E0"

    return np.concatenate([S0.ravel(), E0.ravel(), I0.ravel(), Rec0.ravel()])