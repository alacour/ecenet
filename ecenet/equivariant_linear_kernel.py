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

A Triton program loads one contiguous ``(BLOCK_E, BLOCK_K·n_ang)`` tile of
the interleaved tensor (fully coalesced, read exactly once), splits it in
registers into the ``n_ang`` per-mode ``(BLOCK_E, BLOCK_K)`` slices
(``tl.reshape`` + ``tl.split``), runs the true per-mode dots on tensor cores
into ``n_ang`` accumulators, and joins the results back into one contiguous
interleaved ``(BLOCK_E, BLOCK_N·n_ang)`` store. Cos and sin share a program so
each weight tile is read once for both. No padding, no copies, no per-call
weight assembly. (A first version launched one program per mode with strided
loads, counting on L1 reuse across modes — but those are different programs
on different SMs, so every mode re-read the tile at 25% sector efficiency:
4.5× slower than the dense GEMM on the A100. Hence the register split.)

The grid puts the output-tile index fastest so the programs sharing an edge
panel run adjacently and hit the same L2 lines. ``n_ang`` is padded to a
power of two (≤ 4 supported; larger m_max falls back to the dense path).

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
    ECENET_EL_BLOCK_E/N/K     apply-kernel tiles (default 64 / 32 / 32)
    ECENET_EL_BLOCK_W         dW-kernel out/in tiles (default 32)
    ECENET_EL_WARPS           warps per program (default 8)
    ECENET_EL_STAGES          software-pipelining stages (default 3)
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
_EL_BLOCK_N = int(os.environ.get('ECENET_EL_BLOCK_N', 32))
_EL_BLOCK_K = int(os.environ.get('ECENET_EL_BLOCK_K', 32))
_EL_BLOCK_W = int(os.environ.get('ECENET_EL_BLOCK_W', 32))
_EL_WARPS   = int(os.environ.get('ECENET_EL_WARPS', 8))
_EL_STAGES  = int(os.environ.get('ECENET_EL_STAGES', 3))
_EL_SPLITS  = int(os.environ.get('ECENET_EL_SPLITS', 64))
_EL_MAX_ANG = 4                      # register split/join implemented for NA_PAD ≤ 4


def el_triton_ok(t: torch.Tensor, n_ang: int) -> bool:
    """Dispatch guard: CUDA float32, Triton importable, not disabled, and an
    angular width the register split supports (n_ang ≤ 4, i.e. m_max ≤ 3)."""
    return (_HAS_TRITON and _EL_ENABLED and t.is_cuda
            and t.dtype == torch.float32 and t.numel() > 0
            and 1 <= n_ang <= _EL_MAX_ANG)


def _cdiv(a, b):
    return (a + b - 1) // b


def _na_pad(n_ang):
    return 1 if n_ang == 1 else (2 if n_ang == 2 else 4)


if _HAS_TRITON:

    @triton.jit
    def _split_modes(x, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr, NA: tl.constexpr):
        """(BLOCK_A, BLOCK_B·NA) interleaved tile → NA slices of (BLOCK_A, BLOCK_B).
        Always returns four tensors; slices beyond NA alias slice 0 (unused)."""
        if NA == 1:
            return x, x, x, x
        elif NA == 2:
            x0, x1 = tl.split(tl.reshape(x, (BLOCK_A, BLOCK_B, 2)))
            return x0, x1, x0, x0
        else:
            # flat column k·4+m viewed as (a, b) with m = 2a+b: tl.split peels the
            # LAST axis first → (m0, m2) and (m1, m3), then the a axis.
            xb0, xb1 = tl.split(tl.reshape(x, (BLOCK_A, BLOCK_B, 2, 2)))
            x0, x2 = tl.split(xb0)
            x1, x3 = tl.split(xb1)
            return x0, x1, x2, x3

    @triton.jit
    def _join_modes(y0, y1, y2, y3, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
                    NA: tl.constexpr):
        """Inverse of _split_modes: NA slices → (BLOCK_A, BLOCK_B·NA) interleaved."""
        if NA == 1:
            return y0
        elif NA == 2:
            return tl.reshape(tl.join(y0, y1), (BLOCK_A, BLOCK_B * 2))
        else:
            # inverse order of _split_modes: join(a-pairs) then join along b, so the
            # (a, b) axes flatten to m = 2a+b.
            return tl.reshape(tl.join(tl.join(y0, y2), tl.join(y1, y3)), (BLOCK_A, BLOCK_B * 4))

    @triton.jit
    def _el_apply_kernel(xc_ptr, xs_ptr, w_ptr, b_ptr, yc_ptr, ys_ptr,
                         E, K, N, n_ang,
                         w_stride_m, w_stride_k, w_stride_n,
                         HAS_BIAS: tl.constexpr, TF32: tl.constexpr, NA: tl.constexpr,
                         BLOCK_E: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
        """One (out tile, edge tile), all modes:
        y_m[e, n] = Σ_k x_m[e, k] · Wt_m[k, n] for m < n_ang, cos and sin together.
        Wt_m[k, n] is read at w_ptr + m·w_stride_m + k·w_stride_k + n·w_stride_n,
        so the same kernel serves the forward (k=in, n=out) and dX (k=out, n=in).
        The x tile is one contiguous (BLOCK_E, BLOCK_K·NA) load, split into modes
        in registers; the y tile is joined back and stored contiguously."""
        pid_n = tl.program_id(0)                 # fastest: an edge panel's N tiles
        pid_e = tl.program_id(1)                 # run adjacently and share L2
        offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_a = tl.arange(0, NA)
        e_mask = offs_e < E
        n_mask = offs_n < N
        a_mask = offs_a < n_ang
        offs_e64 = offs_e.to(tl.int64)           # E·K·n_ang can exceed int32

        # x tile (BLOCK_E, BLOCK_K, NA) at e·K·n_ang + k·n_ang + a, flattened to
        # (BLOCK_E, BLOCK_K·NA) row-major (k, a) — contiguous when NA == n_ang.
        x_rows = offs_e64[:, None, None] * (K * n_ang)
        x_cols = offs_k[None, :, None] * n_ang + offs_a[None, None, :]

        acc_c0 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_c1 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_c2 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_c3 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_s0 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_s1 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_s2 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        acc_s3 = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            k_mask = kk < K
            x_off = x_rows + (k0 * n_ang + x_cols)
            x_mask = e_mask[:, None, None] & k_mask[None, :, None] & a_mask[None, None, :]
            xc = tl.reshape(tl.load(xc_ptr + x_off, mask=x_mask, other=0.0),
                            (BLOCK_E, BLOCK_K * NA))
            xs = tl.reshape(tl.load(xs_ptr + x_off, mask=x_mask, other=0.0),
                            (BLOCK_E, BLOCK_K * NA))
            xc0, xc1, xc2, xc3 = _split_modes(xc, BLOCK_E, BLOCK_K, NA)
            xs0, xs1, xs2, xs3 = _split_modes(xs, BLOCK_E, BLOCK_K, NA)
            w_off = kk[:, None] * w_stride_k + offs_n[None, :] * w_stride_n
            w_mask = k_mask[:, None] & n_mask[None, :]
            w0 = tl.load(w_ptr + 0 * w_stride_m + w_off, mask=w_mask, other=0.0)
            if TF32:
                acc_c0 += tl.dot(xc0, w0, input_precision="tf32")
                acc_s0 += tl.dot(xs0, w0, input_precision="tf32")
            else:
                acc_c0 += tl.dot(xc0, w0, input_precision="ieee")
                acc_s0 += tl.dot(xs0, w0, input_precision="ieee")
            if NA >= 2:
                w1 = tl.load(w_ptr + 1 * w_stride_m + w_off,
                             mask=w_mask & (1 < n_ang), other=0.0)
                if TF32:
                    acc_c1 += tl.dot(xc1, w1, input_precision="tf32")
                    acc_s1 += tl.dot(xs1, w1, input_precision="tf32")
                else:
                    acc_c1 += tl.dot(xc1, w1, input_precision="ieee")
                    acc_s1 += tl.dot(xs1, w1, input_precision="ieee")
            if NA >= 4:
                w2 = tl.load(w_ptr + 2 * w_stride_m + w_off,
                             mask=w_mask & (2 < n_ang), other=0.0)
                w3 = tl.load(w_ptr + 3 * w_stride_m + w_off,
                             mask=w_mask & (3 < n_ang), other=0.0)
                if TF32:
                    acc_c2 += tl.dot(xc2, w2, input_precision="tf32")
                    acc_s2 += tl.dot(xs2, w2, input_precision="tf32")
                    acc_c3 += tl.dot(xc3, w3, input_precision="tf32")
                    acc_s3 += tl.dot(xs3, w3, input_precision="tf32")
                else:
                    acc_c2 += tl.dot(xc2, w2, input_precision="ieee")
                    acc_s2 += tl.dot(xs2, w2, input_precision="ieee")
                    acc_c3 += tl.dot(xc3, w3, input_precision="ieee")
                    acc_s3 += tl.dot(xs3, w3, input_precision="ieee")
        if HAS_BIAS:                                   # m=0 cos slot only
            b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
            acc_c0 += b[None, :]
        yc = _join_modes(acc_c0, acc_c1, acc_c2, acc_c3, BLOCK_E, BLOCK_N, NA)
        ys = _join_modes(acc_s0, acc_s1, acc_s2, acc_s3, BLOCK_E, BLOCK_N, NA)
        y_off = tl.reshape(offs_e64[:, None, None] * (N * n_ang)
                           + offs_n[None, :, None] * n_ang + offs_a[None, None, :],
                           (BLOCK_E, BLOCK_N * NA))
        y_mask = tl.reshape(e_mask[:, None, None] & n_mask[None, :, None]
                            & a_mask[None, None, :], (BLOCK_E, BLOCK_N * NA))
        tl.store(yc_ptr + y_off, yc, mask=y_mask)
        tl.store(ys_ptr + y_off, ys, mask=y_mask)

    @triton.jit
    def _el_dw_kernel(xc_ptr, xs_ptr, gc_ptr, gs_ptr, p_ptr,
                      E, Fi, Fo, n_ang, n_i_tiles, chunk,
                      TF32: tl.constexpr, NA: tl.constexpr,
                      BLOCK_E: tl.constexpr, BLOCK_O: tl.constexpr,
                      BLOCK_I: tl.constexpr):
        """dW partial for one (out tile, in tile, edge chunk s), all modes:
        p[s, m, o, i] = Σ_{e in chunk} g_m[e, o]·x_m[e, i]  (cos + sin).
        Contiguous interleaved tile loads, register split per mode. The caller
        sums p over s — a fixed reduction order, no atomics."""
        pid_oi = tl.program_id(0)
        s = tl.program_id(1)
        pid_o = pid_oi // n_i_tiles
        pid_i = pid_oi % n_i_tiles
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_e = tl.arange(0, BLOCK_E)
        offs_a = tl.arange(0, NA)
        o_mask = offs_o < Fo
        i_mask = offs_i < Fi
        a_mask = offs_a < n_ang
        g_cols = offs_o[None, :, None] * n_ang + offs_a[None, None, :]
        x_cols = offs_i[None, :, None] * n_ang + offs_a[None, None, :]
        g_cmask = o_mask[None, :, None] & a_mask[None, None, :]
        x_cmask = i_mask[None, :, None] & a_mask[None, None, :]
        acc0 = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
        e_start = s * chunk
        e_end = tl.minimum(e_start + chunk, E)
        for e0 in range(e_start, e_end, BLOCK_E):
            ee = e0 + offs_e
            e_mask = ee < e_end
            ee64 = ee.to(tl.int64)
            g_off = ee64[:, None, None] * (Fo * n_ang) + g_cols
            x_off = ee64[:, None, None] * (Fi * n_ang) + x_cols
            g_mask = e_mask[:, None, None] & g_cmask
            x_mask = e_mask[:, None, None] & x_cmask
            gc = tl.reshape(tl.load(gc_ptr + g_off, mask=g_mask, other=0.0), (BLOCK_E, BLOCK_O * NA))
            xc = tl.reshape(tl.load(xc_ptr + x_off, mask=x_mask, other=0.0), (BLOCK_E, BLOCK_I * NA))
            gs = tl.reshape(tl.load(gs_ptr + g_off, mask=g_mask, other=0.0), (BLOCK_E, BLOCK_O * NA))
            xs = tl.reshape(tl.load(xs_ptr + x_off, mask=x_mask, other=0.0), (BLOCK_E, BLOCK_I * NA))
            gc0, gc1, gc2, gc3 = _split_modes(gc, BLOCK_E, BLOCK_O, NA)
            xc0, xc1, xc2, xc3 = _split_modes(xc, BLOCK_E, BLOCK_I, NA)
            gs0, gs1, gs2, gs3 = _split_modes(gs, BLOCK_E, BLOCK_O, NA)
            xs0, xs1, xs2, xs3 = _split_modes(xs, BLOCK_E, BLOCK_I, NA)
            if TF32:
                acc0 += tl.dot(tl.trans(gc0), xc0, input_precision="tf32")
                acc0 += tl.dot(tl.trans(gs0), xs0, input_precision="tf32")
                if NA >= 2:
                    acc1 += tl.dot(tl.trans(gc1), xc1, input_precision="tf32")
                    acc1 += tl.dot(tl.trans(gs1), xs1, input_precision="tf32")
                if NA >= 4:
                    acc2 += tl.dot(tl.trans(gc2), xc2, input_precision="tf32")
                    acc2 += tl.dot(tl.trans(gs2), xs2, input_precision="tf32")
                    acc3 += tl.dot(tl.trans(gc3), xc3, input_precision="tf32")
                    acc3 += tl.dot(tl.trans(gs3), xs3, input_precision="tf32")
            else:
                acc0 += tl.dot(tl.trans(gc0), xc0, input_precision="ieee")
                acc0 += tl.dot(tl.trans(gs0), xs0, input_precision="ieee")
                if NA >= 2:
                    acc1 += tl.dot(tl.trans(gc1), xc1, input_precision="ieee")
                    acc1 += tl.dot(tl.trans(gs1), xs1, input_precision="ieee")
                if NA >= 4:
                    acc2 += tl.dot(tl.trans(gc2), xc2, input_precision="ieee")
                    acc2 += tl.dot(tl.trans(gs2), xs2, input_precision="ieee")
                    acc3 += tl.dot(tl.trans(gc3), xc3, input_precision="ieee")
                    acc3 += tl.dot(tl.trans(gs3), xs3, input_precision="ieee")
        p_mask = o_mask[:, None] & i_mask[None, :]
        p_tile = offs_o[:, None] * Fi + offs_i[None, :]
        p_base = p_ptr + (s * n_ang) * Fo * Fi
        tl.store(p_base + p_tile, acc0, mask=p_mask)
        if NA >= 2:
            tl.store(p_base + 1 * Fo * Fi + p_tile, acc1, mask=p_mask & (1 < n_ang))
        if NA >= 4:
            tl.store(p_base + 2 * Fo * Fi + p_tile, acc2, mask=p_mask & (2 < n_ang))
            tl.store(p_base + 3 * Fo * Fi + p_tile, acc3, mask=p_mask & (3 < n_ang))


def _apply(xc, xs, w, bias, K, N, w_strides, tf32):
    """y_m = x_m @ Wt_m for every m. xc/xs: (E, K, n_ang) contiguous fp32;
    w_strides = (stride_m, stride_k, stride_n) of the (K, N) view of each W_m."""
    E, _, na = xc.shape
    yc = torch.empty(E, N, na, dtype=xc.dtype, device=xc.device)
    ys = torch.empty_like(yc)
    grid = (_cdiv(N, _EL_BLOCK_N), _cdiv(E, _EL_BLOCK_E))
    _el_apply_kernel[grid](
        xc, xs, w, bias if bias is not None else w, yc, ys,
        E, K, N, na, *w_strides,
        HAS_BIAS=bias is not None, TF32=tf32, NA=_na_pad(na),
        BLOCK_E=_EL_BLOCK_E, BLOCK_N=_EL_BLOCK_N, BLOCK_K=_EL_BLOCK_K,
        num_warps=_EL_WARPS, num_stages=_EL_STAGES)
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
    n_o, n_i = _cdiv(Fo, _EL_BLOCK_W), _cdiv(Fi, _EL_BLOCK_W)
    grid = (n_o * n_i, n_split)
    _el_dw_kernel[grid](
        xc, xs, gc, gs, partial, E, Fi, Fo, na, n_i, chunk,
        TF32=tf32, NA=_na_pad(na),
        BLOCK_E=_EL_BLOCK_E, BLOCK_O=_EL_BLOCK_W, BLOCK_I=_EL_BLOCK_W,
        num_warps=_EL_WARPS, num_stages=_EL_STAGES)
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
