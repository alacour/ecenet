"""Radial basis functions, cutoff envelopes, and edge/neighbor helpers.

``radial_basis(r, cutoff, n_max, ...)`` expands an interatomic distance into
``n_max`` smooth, cutoff-enveloped features; the ``*_with_deriv`` variants also
return ``df/dr`` for analytic backward. Cutoff functions (cosine / polynomial)
decay smoothly to zero at the cutoff so energies and forces stay continuous as
atoms cross it.

Two-stage cutoff used throughout ECENet:
  1. r_cut_edge:     which atom pairs (i, j) form edges
  2. r_cut_neighbor: for each edge, the atoms k within this distance of either
                     endpoint that enter the ACE atomic basis
"""

import numpy as np
import torch

# =============================================================================
# Cutoff Functions
# =============================================================================

def cosine_cutoff(r, cutoff):
    """Cosine cutoff function (C1 continuous).

    f(r) = 0.5 * (1 + cos(π * r / c))  for r < c, else 0

    Properties:
        - f(0) = 1, f(c) = 0
        - f'(0) = 0, f'(c) = 0
        - f''(c) ≠ 0 (discontinuous second derivative)
    """
    r_scaled = np.pi * r / cutoff
    return torch.where(r < cutoff, 0.5 * (1 + torch.cos(r_scaled)), torch.zeros_like(r))


def cosine_cutoff_with_deriv(r, cutoff):
    """Cosine cutoff function and its derivative."""
    r_scaled = np.pi * r / cutoff
    within = r < cutoff
    fc = torch.where(within, 0.5 * (1 + torch.cos(r_scaled)), torch.zeros_like(r))
    dfc_dr = torch.where(within, -0.5 * (np.pi / cutoff) * torch.sin(r_scaled), torch.zeros_like(r))
    return fc, dfc_dr


def poly_cutoff(r, cutoff):
    """Polynomial cutoff function (C2 continuous).

    f(x) = 1 - 10x³ + 15x⁴ - 6x⁵  where x = r/c, for r < c, else 0

    Properties:
        - f(0) = 1, f(c) = 0
        - f'(0) = 0, f'(c) = 0
        - f''(0) = 0, f''(c) = 0 (continuous second derivative)
    """
    x = r / cutoff
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    fc = 1 - 10*x3 + 15*x4 - 6*x5
    return torch.where(r < cutoff, fc, torch.zeros_like(r))


def poly_cutoff_with_deriv(r, cutoff):
    """Polynomial cutoff function and its derivative.

    f(x) = 1 - 10x³ + 15x⁴ - 6x⁵
    f'(x) = -30x² + 60x³ - 30x⁴ = -30x²(1 - 2x + x²) = -30x²(1-x)²
    df/dr = f'(x) / c
    """
    x = r / cutoff
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    within = r < cutoff

    fc = torch.where(within, 1 - 10*x3 + 15*x4 - 6*x5, torch.zeros_like(r))
    # df/dr = (-30x² + 60x³ - 30x⁴) / c = -30x²(1-x)² / c
    one_minus_x = 1 - x
    dfc_dr = torch.where(within, -30 * x2 * one_minus_x * one_minus_x / cutoff, torch.zeros_like(r))
    return fc, dfc_dr


def get_cutoff_fn(cutoff_type):
    """Get cutoff function by type name."""
    if cutoff_type == 'cosine':
        return cosine_cutoff
    elif cutoff_type == 'poly':
        return poly_cutoff
    else:
        raise ValueError(f"Unknown cutoff type: {cutoff_type}. Use 'cosine' or 'poly'.")


def get_cutoff_fn_with_deriv(cutoff_type):
    """Get cutoff function with derivative by type name."""
    if cutoff_type == 'cosine':
        return cosine_cutoff_with_deriv
    elif cutoff_type == 'poly':
        return poly_cutoff_with_deriv
    else:
        raise ValueError(f"Unknown cutoff type: {cutoff_type}. Use 'cosine' or 'poly'.")


def compute_distance_matrix(positions):
    """Compute pairwise distance matrix.

    Args:
        positions: (n_atoms, 3) atomic positions

    Returns:
        dist_matrix: (n_atoms, n_atoms) pairwise distances
    """
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (n, n, 3)
    return torch.linalg.norm(diff, dim=-1)  # (n, n)


def find_edges(positions, r_cut_edge):
    """Find all edges (atom pairs) within cutoff.

    Args:
        positions: (n_atoms, 3) atomic positions
        r_cut_edge: cutoff distance for edge formation

    Returns:
        edge_i: (n_edges,) tensor of first atom indices
        edge_j: (n_edges,) tensor of second atom indices
    """
    dist_matrix = compute_distance_matrix(positions)
    n_atoms = positions.shape[0]

    # Upper triangular mask (i < j)
    i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=positions.device)
    distances = dist_matrix[i_idx, j_idx]

    # Filter by cutoff
    mask = distances < r_cut_edge
    return i_idx[mask], j_idx[mask]


def radial_basis_with_deriv(r, cutoff, n_max, cutoff_type='cosine'):
    """Radial basis functions and their derivatives.

    f_n(r) = norm * sinc(n*π*r/c) * fc(r)
    where sinc(x) = sin(x)/x and fc(r) is the cutoff function

    Args:
        r: (...) distances
        cutoff: cutoff radius - scalar or tensor broadcastable with r
        n_max: number of basis functions
        cutoff_type: 'cosine' (C1) or 'poly' (C2 continuous)

    Returns:
        f: (..., n_max) basis values
        df_dr: (..., n_max) derivatives w.r.t. r
    """
    # Handle cutoff as tensor or float
    if isinstance(cutoff, torch.Tensor):
        if cutoff.numel() == 1:
            cutoff_val = cutoff.item()
            is_scalar = True
        else:
            cutoff_val = cutoff
            is_scalar = False
    else:
        cutoff_val = float(cutoff)
        is_scalar = True

    # Get the appropriate cutoff function
    cutoff_fn_with_deriv = get_cutoff_fn_with_deriv(cutoff_type)

    n = torch.arange(1, n_max + 1, device=r.device, dtype=r.dtype)

    if is_scalar:
        # Scalar cutoff case
        norm = (2.0 / cutoff_val) ** 0.5

        # x = n*π*r/c
        x = r.unsqueeze(-1) * n * (np.pi / cutoff_val)  # (..., n_max)
        x_safe = x.clamp(min=1e-10)

        # sinc(x) = sin(x)/x, and its derivative
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        sinc = torch.where(x.abs() < 1e-8, torch.ones_like(x), sin_x / x_safe)
        dsinc_dx = torch.where(x.abs() < 1e-8, torch.zeros_like(x), (cos_x - sinc) / x_safe)

        # Cutoff function and derivative
        fc, dfc_dr = cutoff_fn_with_deriv(r, cutoff_val)

        # f = norm * sinc * fc
        f = norm * sinc * fc.unsqueeze(-1)

        # df/dr = norm * (dsinc/dr * fc + sinc * dfc/dr)
        dsinc_dr = dsinc_dx * n * (np.pi / cutoff_val)
        df_dr = norm * (dsinc_dr * fc.unsqueeze(-1) + sinc * dfc_dr.unsqueeze(-1))
    else:
        # Vectorized cutoff case: cutoff_val is (n_samples,)
        c = cutoff_val.unsqueeze(-1)  # (n_samples, 1)
        norm = (2.0 / c) ** 0.5  # (n_samples, 1)

        # x = n*π*r/c  -> (n_samples, n_max)
        x = r.unsqueeze(-1) * n * (np.pi / c)
        x_safe = x.clamp(min=1e-10)

        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        sinc = torch.where(x.abs() < 1e-8, torch.ones_like(x), sin_x / x_safe)
        dsinc_dx = torch.where(x.abs() < 1e-8, torch.zeros_like(x), (cos_x - sinc) / x_safe)

        # Cutoff function
        fc, dfc_dr = cutoff_fn_with_deriv(r, cutoff_val)

        f = norm * sinc * fc.unsqueeze(-1)

        dsinc_dr = dsinc_dx * n * (np.pi / c)
        df_dr = norm * (dsinc_dr * fc.unsqueeze(-1) + sinc * dfc_dr.unsqueeze(-1))

    return f, df_dr


def radial_basis(r, cutoff, n_max, cutoff_type='cosine'):
    """Radial basis functions (same as ACE).

    Args:
        r: (...) distances
        cutoff: cutoff radius - scalar or tensor broadcastable with r
        n_max: number of basis functions
        cutoff_type: 'cosine' (C1) or 'poly' (C2 continuous)

    Returns:
        R: (..., n_max) basis values
    """
    # Handle cutoff as tensor or float
    if isinstance(cutoff, torch.Tensor):
        if cutoff.numel() == 1:
            cutoff_val = cutoff.reshape(())  # scalar tensor
        else:
            cutoff_val = cutoff  # keep as tensor for broadcasting
    else:
        cutoff_val = float(cutoff)

    # Get the appropriate cutoff function
    cutoff_fn = get_cutoff_fn(cutoff_type)

    n = torch.arange(1, n_max + 1, device=r.device, dtype=r.dtype)

    # For vectorized cutoff, we need to broadcast properly
    if isinstance(cutoff_val, torch.Tensor) and cutoff_val.numel() > 1:
        # cutoff_val: (n_samples,), r: (n_samples,)
        # freq: (n_samples, n_max) = cutoff_val.unsqueeze(-1) * n
        freq = n * (np.pi / cutoff_val.unsqueeze(-1))  # (n_samples, n_max)
        norm = (2.0 / cutoff_val.unsqueeze(-1)) ** 0.5  # (n_samples, n_max)

        x = r.unsqueeze(-1) * (n * np.pi / cutoff_val.unsqueeze(-1))  # (n_samples, n_max)
        x_safe = x.clamp(min=1e-10)

        R_raw = torch.where(x.abs() < 1e-8, norm, norm * torch.sin(x) / x_safe)

        # Cutoff function
        fc = cutoff_fn(r, cutoff_val)
        R = R_raw * fc.unsqueeze(-1)
    else:
        # Scalar cutoff - original logic
        if isinstance(cutoff_val, torch.Tensor):
            cutoff_scalar = cutoff_val.item()
        else:
            cutoff_scalar = cutoff_val

        freq = n * (np.pi / cutoff_scalar)
        norm = (2.0 / cutoff_scalar) ** 0.5

        x = r.unsqueeze(-1) * freq
        x_safe = x.clamp(min=1e-10)

        R_raw = torch.where(x.abs() < 1e-8, torch.ones_like(x) * norm, norm * torch.sin(x) / x_safe)

        fc = cutoff_fn(r, cutoff_scalar)
        R = R_raw * fc.unsqueeze(-1)

    return R

