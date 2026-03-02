import torch
import torch.nn as nn
from torch.autograd import Function
import math
from student.flash_attention_backward import flash_attention_backward

class FlashAttentionPyTorch(Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, N_q, d = Q.shape
        _, N_k, _ = K.shape

        # Tile sizes
        B_q = max(16, min(64, N_q))
        B_k = max(16, min(64, N_k))

        scale = 1.0 / math.sqrt(d)

        # Output and logsumexp buffers
        O = torch.zeros(batch, N_q, d, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(batch, N_q, device=Q.device, dtype=torch.float32)

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        for i in range(T_q):
            q_start = i * B_q
            q_end = min(q_start + B_q, N_q)

            # Load Q tile
            Q_i = Q[:, q_start:q_end, :]

            # Running statistics 
            curr_bq = q_end - q_start
            O_i = torch.zeros(batch, curr_bq, d, device=Q.device, dtype=torch.float32)
            l_i = torch.zeros(batch, curr_bq, device=Q.device, dtype=torch.float32)
            m_i = torch.full((batch, curr_bq), float('-inf'), device=Q.device, dtype=torch.float32)

            for j in range(T_k):
                k_start = j * B_k
                k_end = min(k_start + B_k, N_k)

                # Load K, V tiles
                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]

                # Compute attention scores S_i^(j) = Q_i @ K_j^T / sqrt(d)
                # Shape: (batch, curr_bq, curr_bk)
                S_ij = torch.bmm(Q_i.float(), K_j.float().transpose(-1, -2)) * scale

                # Apply causal mask if needed 
                if is_causal:
                
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)  
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)  #
                    mask = k_idx > q_idx  # True where we should mask
                    S_ij = S_ij.masked_fill(mask.unsqueeze(0), -1e6)

                # m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
                # Shape: (batch, curr_bq)
                row_max_new = S_ij.max(dim=-1).values  # (batch, curr_bq)
                m_i_new = torch.maximum(m_i, row_max_new)

                # P_tilde_i^(j) = exp(S_i^(j) - m_i^(j))
                # Shape: (batch, curr_bq, curr_bk)
                P_tilde = torch.exp(S_ij - m_i_new.unsqueeze(-1))

                # l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P_tilde_i^(j))
                correction = torch.exp(m_i - m_i_new)  # (batch, curr_bq)
                l_i_new = correction * l_i + P_tilde.sum(dim=-1)

                # O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) @ O_i^(j-1) + P_tilde_i^(j) @ V_j
                O_i = correction.unsqueeze(-1) * O_i + torch.bmm(P_tilde, V_j.float())

                # Update running stats
                m_i = m_i_new
                l_i = l_i_new

            # Normalize output: O_i = diag(l_i^(T_k))^{-1} @ O_i^(T_k)
            O_i = O_i / l_i.unsqueeze(-1)

            # Logsumexp: L_i = m_i^(T_k) + log(l_i^(T_k))
            L_i = m_i + torch.log(l_i)

            # Write back
            O[:, q_start:q_end, :] = O_i.to(Q.dtype)
            L[:, q_start:q_end] = L_i

        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, is_causal=is_causal)

        return dQ, dK, dV, None