from logging import root

import numpy as np
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
from scipy.integrate import solve_ivp

import src.core.utils as utils
from src.core.SEIRSParameters import SEIRSParameters

def compute_mu_star(params:SEIRSParameters, beta_field, gamma_field):
    n = params.grid_size
    h = params.h
    M = n**2  # total number of nodes in 2D

    # heterogeneous beta and gamma fields and laplacian
    L2D = utils.build_2d_laplacian_neumann(n, n, h)
    beta_field_flat = beta_field.flatten(order='C')
    gamma_field_flat = gamma_field.flatten(order='C')

    e = sp.eye(M)

    # Assemble matrix A (2M x 2M) for the eigenvalue problem
    # System in the form A_block * v = mu * v for positive mu
    A = -bmat(
        [
            [params.dE * L2D - params.sigma*e, diags(beta_field_flat)],
            [params.sigma*e, params.dI * L2D - diags(gamma_field_flat)]
        ],
        format='csr'
    )

    # eigs may return a variety of typed tuples; avoid direct unpacking for type-checkers
    res = eigs(A, k=1, which='LR')  # largest real part
    vals = res[0]
    vecs = res[1]

    mu_star = vals[0].real

    # eigenvector is of length 2M; split into two blocks of length M
    #psi_flat = np.real(vecs[:M, 0])
    #phi_flat = np.real(vecs[M:, 0])

    # normalization so max(psi+phi) = 1
    #max_val = np.max(psi_flat + phi_flat)
    #psi_flat /= max_val
    #phi_flat /= max_val

    return mu_star #, psi_flat, phi_flat


def rhs(t, u, params:SEIRSParameters, beta_field, gamma_field):
    M = params.grid_size ** 2
    S = u[0*M:1*M];  E = u[1*M:2*M]
    I = u[2*M:3*M];  R = u[3*M:4*M]

    L2D = utils.build_2d_laplacian_neumann(params.grid_size, params.grid_size, params.h)
    beta_flat = beta_field.flatten(order='C')
    gamma_flat = gamma_field.flatten(order='C')

    dSdt = params.dS * (L2D @ S) + params.alpha * R   - beta_flat * I * S
    dEdt = params.dE * (L2D @ E) + beta_flat * I * S - params.sigma * E
    dIdt = params.dI * (L2D @ I) + params.sigma * E   - gamma_flat * I
    dRdt = params.dR * (L2D @ R) + gamma_flat * I - params.alpha * R

    return np.concatenate([dSdt, dEdt, dIdt, dRdt])

def rhs_jacobian(t, u, params:SEIRSParameters, beta_field, gamma_field):
    M = params.grid_size ** 2
    S = u[0*M:1*M];  E = u[1*M:2*M]
    I = u[2*M:3*M];  R = u[3*M:4*M]

    L2D = utils.build_2d_laplacian_neumann(params.grid_size, params.grid_size, params.h)
    beta_flat = beta_field.flatten(order='C')
    gamma_flat = gamma_field.flatten(order='C')

    # Diagonal matrices for the nonlinear terms
    diag_S = diags(S)
    diag_I = diags(I)

    # Jacobian blocks
    J11 = params.dS * L2D - diags(beta_flat * I)  # d(dSdt)/dS
    J12 = sp.csr_matrix((M, M))                    # d(dSdt)/dE
    J13 = -diags(beta_flat * S)                    # d(dSdt)/dI
    J14 = params.alpha * sp.eye(M)                 # d(dSdt)/dR

    J21 = diags(beta_flat * I)                     # d(dEdt)/dS
    J22 = params.dE * L2D - params.sigma * sp.eye(M)   # d(dEdt)/dE
    J23 = diags(beta_flat * S)                     # d(dEdt)/dI
    J24 = sp.csr_matrix((M, M))                    # d(dEdt)/dR

    J31 = sp.csr_matrix((M, M))                    # d(dIdt)/dS
    J32 = params.sigma * sp.eye(M)                     # d(dIdt)/dE
    J33 = params.dI * L2D - diags(gamma_flat)      # d(dIdt)/dI
    J34 = sp.csr_matrix((M, M))                    # d(dIdt)/dR

    J41 = sp.csr_matrix((M, M))                    # d(dRdt)/dS
    J42 = sp.csr_matrix((M, M))                    # d(dRdt)/dE
    J43 = diags(gamma_flat)                        # d(dRdt)/dI
    J44 = params.dR * L2D - params.alpha * sp.eye(M)           # d(dRdt)/dR

    # Assemble the full Jacobian matrix
    J = bmat(
        [
            [J11, J12, J13, J14],
            [J21, J22, J23, J24],
            [J31, J32, J33, J34],
            [J41, J42, J43, J44]
        ],
        format='csr'
    )

    return J

def solve_seirs(t_span, t_eval, initial_conditions, params:SEIRSParameters, beta_field, gamma_field):
    return solve_ivp(
        fun=lambda t, y: rhs(t, y, params, beta_field, gamma_field),
        t_span=t_span,
        t_eval=t_eval,
        y0=initial_conditions,
        method='Radau',
        jac=lambda t, y: rhs_jacobian(t, y, params, beta_field, gamma_field),
        rtol=1e-6,
        atol=1e-9
    )

def rhs_bvp_residual(u, params:SEIRSParameters, beta_field, gamma_field):
    n = params.grid_size
    M = n ** 2
    S = u[0*M:1*M];  E = u[1*M:2*M]
    I = u[2*M:3*M];  R = u[3*M:4*M]

    L2D = utils.build_2d_laplacian_neumann(params.grid_size, params.grid_size, params.h)
    beta_flat = beta_field.flatten(order='C')
    gamma_flat = gamma_field.flatten(order='C')

    F1 = params.dS * (L2D @ S) + params.alpha * R   - beta_flat * I * S
    F2 = params.dE * (L2D @ E) + beta_flat * I * S - params.sigma * E
    F3 = params.dI * (L2D @ I) + params.sigma * E   - gamma_flat * I
    F4 = params.dR * (L2D @ R) + gamma_flat * I - params.alpha * R
    F4[-1] = utils.mass(S.reshape(n,n) +
                        E.reshape(n,n) +
                        I.reshape(n,n) +
                        R.reshape(n,n), params.h) - params.N # mass constraint conservation

    return np.concatenate([F1, F2, F3, F4])

def rhs_bvp_jac(u, params: SEIRSParameters, beta_field, gamma_field):
    n = params.grid_size
    M = n ** 2
    S = u[0*M:1*M];  E = u[1*M:2*M]
    I = u[2*M:3*M];  R = u[3*M:4*M]

    L2D = utils.build_2d_laplacian_neumann(params.grid_size, params.grid_size, params.h)
    beta_flat = beta_field.flatten(order='C')
    gamma_flat = gamma_field.flatten(order='C')

    # Jacobian blocks (unchanged)
    J11 = params.dS * L2D - diags(beta_flat * I)
    J12 = sp.csr_matrix((M, M))
    J13 = -diags(beta_flat * S)
    J14 = params.alpha * sp.eye(M)

    J21 = diags(beta_flat * I)
    J22 = params.dE * L2D - params.sigma * sp.eye(M)
    J23 = diags(beta_flat * S)
    J24 = sp.csr_matrix((M, M))

    J31 = sp.csr_matrix((M, M))
    J32 = params.sigma * sp.eye(M)
    J33 = params.dI * L2D - diags(gamma_flat)
    J34 = sp.csr_matrix((M, M))          # d(dIdt)/dR  (comment was wrong, logic was fine)

    J41 = sp.csr_matrix((M, M))
    J42 = sp.csr_matrix((M, M))
    J43 = diags(gamma_flat)
    J44 = params.dR * L2D - params.alpha * sp.eye(M)

    # Assemble as before, but in 'lil' so we can cheaply overwrite one row
    J = bmat(
        [
            [J11, J12, J13, J14],
            [J21, J22, J23, J24],
            [J31, J32, J33, J34],
            [J41, J42, J43, J44]
        ],
        format='lil'
    )

    # --- NEW: mass-constraint row ---
    # rhs_bvp_residual overwrites F4[-1] with g(u) = sum_k w_k*(S_k+E_k+I_k+R_k) - N.
    # Its gradient is the SAME quadrature weight vector w.r.t. every compartment.
    # Previously this row still held the gradient of the discarded pointwise
    # R-equation (J41=J42=0 structurally, J43/J44 untouched) -- inconsistent
    # with the residual, which is what was stalling Newton.
    w = _trapz_weights_2d(n, params.h)     # length-M quadrature weights, order='C'
    last = 4 * M - 1
    J[last, :] = 0.0
    J[last, 0*M:1*M] = w
    J[last, 1*M:2*M] = w
    J[last, 2*M:3*M] = w
    J[last, 3*M:4*M] = w

    return J.tocsr().toarray()

def _trapz_weights_2d(n, h):
    """2D nested-trapezoidal quadrature weights, flattened order='C',
    matching np.trapz(np.trapz(f, dx=h, axis=1), dx=h)."""
    w1d = np.full(n, h)
    w1d[0] *= 0.5
    w1d[-1] *= 0.5
    return np.outer(w1d, w1d).flatten(order='C')

