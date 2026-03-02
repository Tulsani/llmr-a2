import math
import torch
import triton
import triton.language as tl
from torch.autograd import Function


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
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
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


    Q_tile = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)


    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_acc = tl.zeros((Q_TILE_SIZE,),   dtype=tl.float32)
    m_acc = tl.full( (Q_TILE_SIZE,),   float("-inf"), dtype=tl.float32)

    # Query positions for causal masking
    q_start = query_tile_index * Q_TILE_SIZE
    q_offs  = q_start + tl.arange(0, Q_TILE_SIZE)   # (Q_TILE_SIZE,)

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)


    for j in range(T_k):

        K_tile = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
        V_tile = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

        # S_ij = Q_i @ K_j^T * scale  — shape (Q_TILE_SIZE, K_TILE_SIZE)
        S = tl.dot(Q_tile, tl.trans(K_tile)) * scale   # float32 accumulation

        # Causal mask: mask out positions where k_pos > q_pos
        if IS_CAUSAL:
            k_offs = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)   # (K_TILE_SIZE,)
            # Broadcast: (Q_TILE_SIZE, 1) vs (1, K_TILE_SIZE)
            causal_mask = q_offs[:, None] >= k_offs[None, :]
            S = tl.where(causal_mask, S, -1e6)

        # m_i^(j) = max(m_i^(j-1), rowmax(S))
        row_max = tl.max(S, axis=1)                        # (Q_TILE_SIZE,)
        m_new   = tl.maximum(m_acc, row_max)               # (Q_TILE_SIZE,)

        # P_tilde = exp(S - m_new)  — (Q_TILE_SIZE, K_TILE_SIZE)
        P_tilde = tl.exp(S - m_new[:, None])

        # Correction factor for previously accumulated values
        correction = tl.exp(m_acc - m_new)                 # (Q_TILE_SIZE,)

        # l_i^(j) = correction * l_i^(j-1) + rowsum(P_tilde)
        l_acc = correction * l_acc + tl.sum(P_tilde, axis=1)

        # O_i^(j) = diag(correction) @ O_i^(j-1) + P_tilde @ V_j
        # Cast P_tilde to V's dtype before the matmul, accumulate into float32
        O_acc = correction[:, None] * O_acc + tl.dot(
            P_tilde.to(V_tile.dtype), V_tile, acc=tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        )

        # Update running max
        m_acc = m_new

        # Advance K and V pointers to the next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    O_acc = O_acc / l_acc[:, None]                         # (Q_TILE_SIZE, D)
    L_out = m_acc + tl.log(l_acc)                          # (Q_TILE_SIZE,)

    # Write outputs, casting O back to the original dtype
    tl.store(O_block_ptr, O_acc.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_out)




class FlashAttentionTriton(Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        """
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "inputs must be on CUDA"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        batch, N_q, d = Q.shape
        _,     N_k, _ = K.shape

        # Tile sizes — powers of 2, at least 16
        Q_TILE_SIZE = max(16, min(64, triton.next_power_of_2(N_q)))
        K_TILE_SIZE = max(16, min(64, triton.next_power_of_2(N_k)))

        scale = 1.0 / math.sqrt(d)
        T_q   = math.ceil(N_q / Q_TILE_SIZE)

        O = torch.empty_like(Q)
        L = torch.empty(batch, N_q, device=Q.device, dtype=torch.float32)

        # Launch grid: (T_q, batch)
        grid = (T_q, batch)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass not yet implemented")