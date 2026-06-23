"""ACE atomic basis for ECENet, as a custom autograd Function.

This module is the geometric front-end of the model: it builds the per-atom
ACE descriptor

    A[atom, neighbor_type, n, (l, m)] = Σ_neighbors  f_n(r) · Y_lm(r_hat)

i.e. a radial basis ``f_n`` (rank ``n_max``) outer-product the real spherical
harmonics ``Y_lm``, summed over neighbors and split by neighbor type. Forward
output shape: ``(B, N, n_types, n_max, n_sph)`` with ``n_sph = (l_max+1)**2``.

Everything downstream lives in ``model.py``: rotation of these features into
each bond frame (``wigner_rotate``), the reshape to the ``(A_cos, A_sin)``
SO(2) feature pairs, the equivariant layers, and the readout.

``ACEBasisAnalytic`` is a ``torch.autograd.Function`` whose backward uses the
analytic spherical-harmonic Jacobians from sphericart, stored as plain
(grad-free) tensors. This keeps the second-order backward needed for force
training cheap — it never re-traverses the spherical-harmonic graph.
"""

import torch

from ecenet.radial import radial_basis, radial_basis_with_deriv
from ecenet.spherical import _get_sph_cache


def _get_sph(l_max: int):
    """Return (cached) SphericalHarmonics object for the given l_max."""
    return _get_sph_cache(l_max)


class ACEBasisAnalytic(torch.autograd.Function):
    """Vectorized ACE atomic basis with analytic SH Jacobians from sphericart.

    Wraps _compute_atomic_basis_batch as a custom Function that stores
    dY/dxyz from sphericart.compute_with_gradients as plain tensors (no
    grad_fn).  This eliminates the expensive SH graph re-traversal during
    the 2nd-order backward (force training).

    Forward output:
        A: (B, N, n_types, n_max, n_sph)

    The backward uses stored plain Jacobians for the SH contribution and
    recomputes df_R/dr via autograd on r only (cheap — no SH graph).
    """

    @staticmethod
    def forward(ctx, pos_batch, nb_src, nb_dst, types,
                r_cut, n_max, l_max, n_types, cutoff_type_id,
                shift_vecs_nb=None):
        B, N, _ = pos_batch.shape
        n_nb    = nb_src.shape[0]
        n_sph   = (l_max + 1) ** 2
        device, dtype = pos_batch.device, pos_batch.dtype
        cutoff_type = ('cosine', 'poly')[cutoff_type_id]

        if n_nb == 0:
            A = torch.zeros(B, N, n_types, n_max, n_sph, device=device, dtype=dtype)
            ctx.meta = (nb_src, nb_dst, types, r_cut, n_max, l_max,
                            n_types, cutoff_type, B, N, n_sph, 0)
            ctx.save_for_backward()
            return A

        # Neighbor displacement vectors: (B, n_nb, 3)
        diff = pos_batch[:, nb_dst] - pos_batch[:, nb_src]        # (B, n_nb, 3)
        if shift_vecs_nb is not None:
            diff = diff + shift_vecs_nb.to(dtype=dtype)[None, :, :]  # PBC offset
        r = torch.sqrt((diff ** 2).sum(-1) + 1e-30)            # (B, n_nb)

        # Radial basis: (B, n_nb, n_max)
        f_R = radial_basis(r.reshape(-1), r_cut, n_max,
                           cutoff_type=cutoff_type).reshape(B, n_nb, n_max)

        # Sphericart SH + analytic Jacobians — plain tensors (no grad_fn)
        sph     = _get_sph(l_max)
        diff_flat = diff.detach().reshape(-1, 3).contiguous()
        Y_flat, dYdxyz_flat = sph.compute_with_gradients(diff_flat)
        # Y_flat: (B*n_nb, n_sph),  dYdxyz_flat: (B*n_nb, 3, n_sph)
        Y       = Y_flat.to(dtype=dtype).reshape(B, n_nb, n_sph)
        dYdxyz  = dYdxyz_flat.to(dtype=dtype).reshape(B, n_nb, 3, n_sph)

        # Contributions: (B, n_nb, n_max, n_sph)
        contributions = f_R.unsqueeze(-1) * Y.unsqueeze(-2)

        # Scatter into (B, N*n_types, n_max, n_sph)
        neighbor_types = types[nb_dst]                             # (n_nb,)
        flat_idx       = nb_src * n_types + neighbor_types         # (n_nb,)
        flat_idx_exp   = flat_idx[None, :, None, None].expand(B, n_nb, n_max, n_sph)
        A_flat = torch.zeros(B, N * n_types, n_max, n_sph, device=device, dtype=dtype)
        A_flat = A_flat.scatter_add(1, flat_idx_exp, contributions)
        A      = A_flat.reshape(B, N, n_types, n_max, n_sph)

        ctx.save_for_backward(diff, r, f_R, dYdxyz, Y)
        ctx.meta = (nb_src, nb_dst, types, r_cut, n_max, l_max,
                        n_types, cutoff_type, B, N, n_sph, n_nb)
        return A

    @staticmethod
    def backward(ctx, grad_A):
        if not ctx.saved_tensors:
            return (None,) * 10
        diff, r, f_R, dYdxyz, Y = ctx.saved_tensors
        (nb_src, nb_dst, types, r_cut, n_max, l_max,
         n_types, cutoff_type, B, N, n_sph, n_nb) = ctx.meta

        device, dtype = diff.device, diff.dtype

        # Gather grad_A for each neighbor pair → grad_contrib: (B, n_nb, n_max, n_sph)
        neighbor_types = types[nb_dst]
        flat_idx       = nb_src * n_types + neighbor_types
        grad_A_flat    = grad_A.reshape(B, N * n_types, n_max, n_sph)
        grad_contrib   = grad_A_flat[:, flat_idx]                  # (B, n_nb, n_max, n_sph)

        # grad_Y: (B, n_nb, n_sph) — sum over n_max dimension
        grad_Y = (f_R.unsqueeze(-1) * grad_contrib).sum(-2)        # (B, n_nb, n_sph)

        # grad_diff from SH Jacobian: dYdxyz is plain (no grad_fn in 2nd-order bwd)
        # dYdxyz: (B, n_nb, 3, n_sph) = dY_s/d(diff_x)
        # grad_diff_Y[b,k,x] = Σ_s dYdxyz[b,k,x,s] * grad_Y[b,k,s]
        grad_diff_Y = torch.einsum('bks,bkxs->bkx', grad_Y, dYdxyz)  # (B, n_nb, 3)

        # grad_diff from radial basis: use analytic df_R/dr (plain tensor, no graph)
        grad_f_R    = (grad_contrib * Y.unsqueeze(-2)).sum(-1)       # (B, n_nb, n_max)
        _, df_R_dr  = radial_basis_with_deriv(r.detach().reshape(-1), r_cut, n_max,
                                             cutoff_type=cutoff_type)
        df_R_dr     = df_R_dr.reshape(B, n_nb, n_max)
        grad_r      = (df_R_dr * grad_f_R).sum(-1)                   # (B, n_nb)
        grad_diff_r = (diff / r.unsqueeze(-1)) * grad_r.unsqueeze(-1)  # (B, n_nb, 3)

        grad_diff_total = grad_diff_Y + grad_diff_r                # (B, n_nb, 3)

        # Scatter back to (B, N, 3) using non-in-place ops for 2nd-order autograd support
        zeros   = torch.zeros(B, N, 3, device=device, dtype=dtype)
        dst_exp = nb_dst[None, :, None].expand(B, n_nb, 3)
        src_exp = nb_src[None, :, None].expand(B, n_nb, 3)
        grad_pos = (zeros.scatter_add(1, dst_exp,  grad_diff_total)
                  + zeros.scatter_add(1, src_exp, -grad_diff_total))

        grad_shift = grad_diff_total.sum(0) if ctx.needs_input_grad[9] else None
        return grad_pos, None, None, None, None, None, None, None, None, grad_shift
