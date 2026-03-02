import torch
import math
import sys


@torch.compile
def flash_attention_backward(Q, K, V, O, dO, L, is_causal=False):
    """
    FlashAttention-2 backward pass
    """
    batch, N_q, d = Q.shape
    scale = 1.0 / math.sqrt(d)


    D = (O * dO).sum(dim=-1)   

    S = torch.bmm(Q.float(), K.float().transpose(-1, -2)) * scale  

    
    if is_causal:
        q_idx = torch.arange(N_q, device=Q.device).unsqueeze(1)   # (N_q, 1)
        k_idx = torch.arange(K.shape[1], device=K.device).unsqueeze(0)  # (1, N_k)
        causal_mask = k_idx > q_idx                                # (N_q, N_k)
        S = S.masked_fill(causal_mask.unsqueeze(0), float('-inf'))


    P = torch.exp(S - L.unsqueeze(-1).float())   # (batch, N_q, N_k)


    dV = torch.bmm(P.transpose(-1, -2), dO.float())   # (batch, N_k, d)


    dP = torch.bmm(dO.float(), V.float().transpose(-1, -2))


    dS = P * (dP - D.unsqueeze(-1).float())   # (batch, N_q, N_k)


    dQ = torch.bmm(dS, K.float()) * scale


    dK = torch.bmm(dS.transpose(-1, -2), Q.float()) * scale


    orig_dtype = Q.dtype
    return dQ.to(orig_dtype), dK.to(orig_dtype), dV.to(orig_dtype)

if torch.cuda.is_available():
    flash_attention_backward = torch.compile(flash_attention_backward)