"""ecenet/equivariant_linear_kernel.py — block-diagonal EquivariantLinear on Triton.

``EquivariantLinear`` computes, per angular mode m,

    y[e, o, m] = Σ_i x[e, i, m] · W[m, o, i]        (+ bias[o] on the m=0 cos slot)

with the activations stored interleaved as ``(n_edges, features, n_ang)``. The
per-m slice ``x[:, :, m]`` has no unit-stride axis (stride n_ang along the
feature axis), so cuBLAS cannot take it as a batched GEMM without a copy. The
two existing paths pay for that in different ways:

* einsum — lowers to a batched bmm over m, copying every activation into an
  m-major layout and emitting m-major output (layout churn, ~29% of the step
  when it was the default);
* dense — one row-major GEMM against an ``(in·n_ang, out·n_ang)`` weight that
  is block-diagonal per m. No copies, but the zero blocks cost n_ang× the
  strictly needed flops (4× at m_max=3): on the 512-atom diamond box each
  down/up projection ran ~1.2 ms where the strict arithmetic is ~0.3 ms.

A Triton program reads a strided per-m tile straight from the interleaved
tensor (the four modes' loads of one edge tile hit the same L1 lines, so the
DRAM traffic stays ~1×), runs the true ``(BLOCK_E, K) @ (K, BLOCK_N)`` dot for
that mode on tensor cores, and stores the result back interleaved. Cos and sin
share a program so each weight tile is read once for both. No padding, no
copies, no per-call weight assembly.

Precision: ``tl.dot`` follows ``torch.backends.cuda.matmul.allow_tf32`` — TF32
when the process has it on (matching what the dense GEMM it replaces did),
IEEE fp32 otherwise. This deviates from the edge-frame/real-space kernels'
"ieee always" rule on purpose: the reference here is itself a TF32 GEMM.

Backward (``EquivariantLinearTriton``): dX is the same kernel with the weight
read transposed; dW contracts over edges, computed as per-edge-chunk partials
by a second kernel and summed in torch (deterministic — no atomics); dbias is
a torch reduction. dW is skipped when the weights do not require grad, so
inference-time force evaluation (``autograd.grad(E, pos)``) pays only the dX
GEMM — the calculator freezes parameters for exactly this reason.

Env knobs (read once at import):
    ECENET_EL_TRITON=0        disable (auto-dispatch falls back to dense/einsum)
    ECENET_EL_BLOCK_E/N/K     tile sizes (default 64 / 64 / 32)
    ECENET_EL_WARPS           warps per program (default 4)
    ECENET_EL_SPLITS          max edge splits for the dW partials (default 64)
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:                       # CPU-only / no-triton env → torch paths
    _HAS_TRITON = False

_EL_ENABLED = os.environ.get('ECENET_EL_TRITON', '1') == '1'
_EL_BLOCK_E = int(os.environ.get('ECENET_EL_BLOCK_E', 64))
_EL_BLOCK_N = int(os.environ.get('ECENET_EL_BLOCK_N', 64))
_EL_BLOCK_K = int(os.environ.get('ECENET_EL_BLOCK_K', 32))
_EL_WARPS   = int(os.environ.get('ECENET_EL_WARPS', 4))
_EL_SPLITS  = int(os.environ.get('ECENET_EL_SPLITS', 64))


def el_triton_ok(t: torch.Tensor) -> bool:
    """Dispatch guard: CUDA float32 with Triton importable and not disabled."""
    return (_HAS_TRITON and _EL_ENABLED and t.is_cuda
            and t.dtype == torch.float32 and t.numel() > 0)


def _cdiv(a, b):
    return (a + b - 1) // b


if _HAS_TRITON:

    @triton.jit
    def _el_apply_kernel(xc_ptr, xs_ptr, w_ptr, b_ptr, yc_ptr, ys_ptr,
                         E, K, N, n_ang,
                         w_stride_m, w_stride_k, w_stride_n,
                         HAS_BIAS: tl.constexpr, TF32: tl.constexpr,
                         BLOCK_E: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
        """One (edge tile, out tile, mode m):
        y_m[e, n] = Σ_k x_m[e, k] · Wt_m[k, n], cos and sin halves together.
        Wt_m[k, n] is read at w_ptr + m·w_stride_m + k·w_stride_k + n·w_stride_n,
        so the same kernel serves the forward (k=in, n=out) and dX (k=out, n=in)."""
        pid_e = tl.program_id(0)
        pid_n = tl.program_id(1)
        m = tl.program_id(2)
        offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        e_mask = offs_e < E
        n_mask = offs_n < N

        # Interleaved (E, K, n_ang): element (e, k, m) at e·K·n_ang + k·n_ang + m.
        offs_e64 = offs_e.to(tl.int64)             # E·K·n_ang can exceed int32
        x_rows = offs_e64[:, None] * (K * n_ang) + m
        w_base = w_ptr + m * w_stride_m
        acc_c = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_s = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            k_mask = kk < K
            x_off = x_rows + kk[None, :] * n_ang
            x_mask = e_mask[:, None] & k_mask[None, :]
            xc = tl.load(xc_ptr + x_off, mask=x_mask, other=0.0)
            xs = tl.load(xs_ptr + x_off, mask=x_mask, other=0.0)
            w = tl.load(w_base + kk[:, None] * w_stride_k + offs_n[None, :] * w_stride_n,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            if TF32:
                acc_c += tl.dot(xc, w, input_precision="tf32")
                acc_s += tl.dot(xs, w, input_precision="tf32")
            else:
                acc_c += tl.dot(xc, w, input_precision="ieee")
                acc_s += tl.dot(xs, w, input_precision="ieee")
        if HAS_BIAS:                                   # m=0 cos slot only
            b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
            b = tl.where(m == 0, b, 0.0)
            acc_c += b[None, :]
        y_off = offs_e64[:, None] * (N * n_ang) + offs_n[None, :] * n_ang + m
        y_mask = e_mask[:, None] & n_mask[None, :]
        tl.store(yc_ptr + y_off, acc_c, mask=y_mask)
        tl.store(ys_ptr + y_off, acc_s, mask=y_mask)

    @triton.jit
    def _el_dw_kernel(xc_ptr, xs_ptr, gc_ptr, gs_ptr, p_ptr,
                      E, Fi, Fo, n_ang, n_i_tiles, chunk,
                      TF32: tl.constexpr,
                      BLOCK_E: tl.constexpr, BLOCK_O: tl.constexpr,
                      BLOCK_I: tl.constexpr):
        """dW partial for one (out tile, in tile, mode m, edge chunk s):
        p[s, m, o, i] = Σ_{e in chunk} g_m[e, o]·x_m[e, i]  (cos + sin).
        The caller sums p over s — a fixed reduction order, no atomics."""
        pid_oi = tl.program_id(0)
        m = tl.program_id(1)
        s = tl.program_id(2)
        pid_o = pid_oi // n_i_tiles
        pid_i = pid_oi % n_i_tiles
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_e = tl.arange(0, BLOCK_E)
        o_mask = offs_o < Fo
        i_mask = offs_i < Fi
        acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
        e_start = s * chunk
        e_end = tl.minimum(e_start + chunk, E)
        for e0 in range(e_start, e_end, BLOCK_E):
            ee = e0 + offs_e
            e_mask = ee < e_end
            ee64 = ee.to(tl.int64)
            g_off = ee64[:, None] * (Fo * n_ang) + offs_o[None, :] * n_ang + m
            x_off = ee64[:, None] * (Fi * n_ang) + offs_i[None, :] * n_ang + m
            g_mask = e_mask[:, None] & o_mask[None, :]
            x_mask = e_mask[:, None] & i_mask[None, :]
            gc = tl.load(gc_ptr + g_off, mask=g_mask, other=0.0)
            xc = tl.load(xc_ptr + x_off, mask=x_mask, other=0.0)
            gs = tl.load(gs_ptr + g_off, mask=g_mask, other=0.0)
            xs = tl.load(xs_ptr + x_off, mask=x_mask, other=0.0)
            if TF32:
                acc += tl.dot(tl.trans(gc), xc, input_precision="tf32")
                acc += tl.dot(tl.trans(gs), xs, input_precision="tf32")
            else:
                acc += tl.dot(tl.trans(gc), xc, input_precision="ieee")
                acc += tl.dot(tl.trans(gs), xs, input_precision="ieee")
        p_off = ((s * n_ang + m) * Fo + offs_o[:, None]) * Fi + offs_i[None, :]
        tl.store(p_ptr + p_off, acc, mask=o_mask[:, None] & i_mask[None, :])


def _apply(xc, xs, w, bias, K, N, w_strides, tf32):
    """y_m = x_m @ Wt_m for every m. xc/xs: (E, K, n_ang) contiguous fp32;
    w_strides = (stride_m, stride_k, stride_n) of the (K, N) view of each W_m."""
    E, _, na = xc.shape
    yc = torch.empty(E, N, na, dtype=xc.dtype, device=xc.device)
    ys = torch.empty_like(yc)
    grid = (_cdiv(E, _EL_BLOCK_E), _cdiv(N, _EL_BLOCK_N), na)
    _el_apply_kernel[grid](
        xc, xs, w, bias if bias is not None else w, yc, ys,
        E, K, N, na, *w_strides,
        HAS_BIAS=bias is not None, TF32=tf32,
        BLOCK_E=_EL_BLOCK_E, BLOCK_N=_EL_BLOCK_N, BLOCK_K=_EL_BLOCK_K,
        num_warps=_EL_WARPS)
    return yc, ys


def _dweight(xc, xs, gc, gs, tf32):
    """dW[m, o, i] = Σ_e g[e, o, m]·x[e, i, m] over cos and sin: chunked
    partials over edges, summed in torch."""
    E, Fi, na = xc.shape
    Fo = gc.shape[1]
    n_tiles_e = _cdiv(E, _EL_BLOCK_E)
    n_split = max(1, min(_EL_SPLITS, n_tiles_e))
    chunk = _cdiv(n_tiles_e, n_split) * _EL_BLOCK_E
    n_split = _cdiv(E, chunk)                  # no empty trailing splits
    partial = torch.empty(n_split, na, Fo, Fi, dtype=xc.dtype, device=xc.device)
    n_o, n_i = _cdiv(Fo, _EL_BLOCK_N), _cdiv(Fi, _EL_BLOCK_N)
    grid = (n_o * n_i, na, n_split)
    _el_dw_kernel[grid](
        xc, xs, gc, gs, partial, E, Fi, Fo, na, n_i, chunk,
        TF32=tf32, BLOCK_E=_EL_BLOCK_E, BLOCK_O=_EL_BLOCK_N, BLOCK_I=_EL_BLOCK_N,
        num_warps=_EL_WARPS)
    return partial.sum(0)


class EquivariantLinearTriton(torch.autograd.Function):
    """y[..., o, m] = Σ_i x[..., i, m]·W[m, o, i] (+ bias on the m=0 cos slot)
    via the per-mode Triton dot; analytic backward (dX same kernel with the
    weight transposed; dW chunked partials; dbias torch)."""

    @staticmethod
    def forward(ctx, A_cos, A_sin, weights, bias):
        na, Fo, Fi = weights.shape
        lead = A_cos.shape[:-2]
        xc = A_cos.reshape(-1, Fi, na).contiguous()
        xs = A_sin.reshape(-1, Fi, na).contiguous()
        w = weights.contiguous()
        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        # Forward view of W_m as (K=in, N=out): Wt[k, n] = W[m, n, k].
        yc, ys = _apply(xc, xs, w, bias.contiguous(), Fi, Fo,
                        (w.stride(0), w.stride(2), w.stride(1)), tf32)
        ctx.save_for_backward(xc, xs, w)
        ctx.tf32, ctx.lead = tf32, lead
        return yc.view(*lead, Fo, na), ys.view(*lead, Fo, na)

    @staticmethod
    def backward(ctx, g_cos, g_sin):
        xc, xs, w = ctx.saved_tensors
        na, Fo, Fi = w.shape
        gc = g_cos.reshape(-1, Fo, na).contiguous()
        gs = g_sin.reshape(-1, Fo, na).contiguous()
        dA_cos = dA_sin = dW = db = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # dX_m = g_m @ W_m: view W_m as (K=out, N=in): Wt[k, n] = W[m, k, n].
            dxc, dxs = _apply(gc, gs, w, None, Fo, Fi,
                              (w.stride(0), w.stride(1), w.stride(2)), ctx.tf32)
            dA_cos = dxc.view(*ctx.lead, Fi, na)
            dA_sin = dxs.view(*ctx.lead, Fi, na)
        if ctx.needs_input_grad[2]:
            dW = _dweight(xc, xs, gc, gs, ctx.tf32)
        if ctx.needs_input_grad[3]:
            db = gc[:, :, 0].sum(0)
        return dA_cos, dA_sin, dW, db
