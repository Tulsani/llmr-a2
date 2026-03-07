import math
import torch
import triton
import triton.language as tl
from triton import cdiv
from torch.autograd import Function

from student.flash_attention_backward import flash_attention_backward


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    N_K_TILES: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)

    for j in range(N_K_TILES):
        k_start = j * K_TILE_SIZE

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Qi.to(tl.float32), tl.trans(Kj.to(tl.float32)), input_precision="ieee") * scale

        if IS_CAUSAL:
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            Sij = tl.where(causal_mask, Sij, -1e6)

        mij = tl.max(Sij, axis=1)
        mi_new = tl.maximum(mi, mij)
        Pij = tl.exp(Sij - mi_new[:, None])
        correction = tl.exp(mi - mi_new)

        li = correction * li + tl.sum(Pij, axis=1)
        Oi = correction[:, None] * Oi + tl.dot(Pij.to(tl.float32), Vj.to(tl.float32), input_precision="ieee")
        mi = mi_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0,))


def _pytorch_tiled_forward(Q, K, V, is_causal):
    """
    Calls FlashAttentionPyTorch's forward logic directly via a no_grad forward
    """
    batch, N_q, d = Q.shape
    _, N_k, _ = K.shape
    B_q = max(16, min(64, N_q))
    B_k = max(16, min(64, N_k))
    scale = 1.0 / math.sqrt(d)
    T_q = math.ceil(N_q / B_q)
    T_k = math.ceil(N_k / B_k)

    O = torch.zeros(batch, N_q, d, device=Q.device, dtype=Q.dtype)
    L = torch.zeros(batch, N_q, device=Q.device, dtype=torch.float32)

    for i in range(T_q):
        q_start = i * B_q
        q_end = min(q_start + B_q, N_q)
        Q_i = Q[:, q_start:q_end, :]
        curr_bq = q_end - q_start
        O_i = torch.zeros(batch, curr_bq, d, device=Q.device, dtype=torch.float32)
        l_i = torch.zeros(batch, curr_bq, device=Q.device, dtype=torch.float32)
        m_i = torch.full((batch, curr_bq), float('-inf'), device=Q.device, dtype=torch.float32)

        for j in range(T_k):
            k_start = j * B_k
            k_end = min(k_start + B_k, N_k)
            K_j = K[:, k_start:k_end, :]
            V_j = V[:, k_start:k_end, :]
            S_ij = torch.bmm(Q_i.float(), K_j.float().transpose(-1, -2)) * scale
            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                S_ij = S_ij.masked_fill((k_idx > q_idx).unsqueeze(0), -1e6)
            m_i_new = torch.maximum(m_i, S_ij.max(dim=-1).values)
            P_tilde = torch.exp(S_ij - m_i_new.unsqueeze(-1))
            correction = torch.exp(m_i - m_i_new)
            l_i = correction * l_i + P_tilde.sum(dim=-1)
            O_i = correction.unsqueeze(-1) * O_i + torch.bmm(P_tilde, V_j.float())
            m_i = m_i_new

        O_i = O_i / l_i.unsqueeze(-1)
        O[:, q_start:q_end, :] = O_i.to(Q.dtype)
        L[:, q_start:q_end] = m_i + torch.log(l_i)

    return O, L


class FlashAttentionTriton(Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        if not Q.is_cuda:
            O, L = _pytorch_tiled_forward(Q, K, V, is_causal)
            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal
            return O

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        batch, N_q, d = Q.shape
        _,     N_k, _ = K.shape

        Q_TILE_SIZE = max(16, min(64, triton.next_power_of_2(N_q)))
        K_TILE_SIZE = max(16, min(64, triton.next_power_of_2(N_k)))
        n_q_tiles   = cdiv(N_q, Q_TILE_SIZE)
        n_k_tiles   = cdiv(N_k, K_TILE_SIZE)

        scale = 1.0 / math.sqrt(d)

        O = torch.empty_like(Q)
        L = torch.empty(batch, N_q, device=Q.device, dtype=Q.dtype)

        flash_fwd_kernel[(n_q_tiles, batch)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_q,
            N_KEYS=N_k,
            scale=scale,
            IS_CAUSAL=is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            N_K_TILES=n_k_tiles,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, is_causal=ctx.is_causal)
        return dQ, dK, dV, None