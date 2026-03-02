import math
import itertools
import torch
import triton
import triton.testing
import pandas as pd


from student.flash_attention_pytroch import FlashAttentionPyTorch
from student.flash_attention_titron import FlashAttentionTriton



def pytorch_attention(Q, K, V, is_causal=True):
    """Standard scaled dot-product attention — materializes full N x N matrix."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.bmm(Q, K.transpose(-1, -2)) * scale          # (B, N_q, N_k)
    if is_causal:
        N_q, N_k = Q.shape[1], K.shape[1]
        mask = torch.triu(
            torch.ones(N_q, N_k, device=Q.device, dtype=torch.bool), diagonal=1
        )
        S = S.masked_fill(mask.unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    return torch.bmm(P, V)


def make_inputs(seq_len, d, dtype, device='cuda'):
    shape = (1, seq_len, d)
    Q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


def pick_tile_sizes(seq_len, d):

    tile = max(16, min(128, triton.next_power_of_2(seq_len // 16)))
    return tile, tile


def bench_forward(fn, Q, K, V):
    """Benchmark forward pass only."""
    return triton.testing.do_bench(
        lambda: fn(Q, K, V, True),
        warmup=25, rep=100
    )


def bench_backward(fn, Q, K, V):
    """Benchmark backward pass only (excludes forward time)."""
    # Pre-run forward outside the timed region
    O = fn(Q, K, V, True)
    dO = torch.ones_like(O)

    def bwd():
        # Zero grads each rep so accumulation doesn't affect timing
        if Q.grad is not None: Q.grad.zero_()
        if K.grad is not None: K.grad.zero_()
        if V.grad is not None: V.grad.zero_()
        O.backward(dO, retain_graph=True)

    return triton.testing.do_bench(bwd, warmup=25, rep=100)


def bench_fwd_bwd(fn, Q, K, V):
    """Benchmark end-to-end forward + backward."""
    dO_store = [None]

    def fwd_bwd():
        if Q.grad is not None: Q.grad.zero_()
        if K.grad is not None: K.grad.zero_()
        if V.grad is not None: V.grad.zero_()
        O = fn(Q, K, V, True)
        dO = torch.ones_like(O)
        O.backward(dO)

    return triton.testing.do_bench(fwd_bwd, warmup=25, rep=100)


def run_benchmark():
    device = 'cuda'
    assert torch.cuda.is_available(), "CUDA required"

    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_sizes     = [16, 32, 64, 128]
    precisions  = [torch.bfloat16, torch.float32]

    flash_apply  = FlashAttentionTriton.apply
    pytorch_apply = pytorch_attention

    rows = []

    for dtype, seq_len, d in itertools.product(precisions, seq_lengths, d_sizes):
        dtype_name = 'bf16' if dtype == torch.bfloat16 else 'fp32'
        print(f"  dtype={dtype_name:4s}  seq={seq_len:6d}  d={d:3d}", end="  ", flush=True)

        Q, K, V = make_inputs(seq_len, d, dtype, device)

        # ---- FlashAttention-2 (Triton) --------------------------------
        try:
            fwd_flash  = bench_forward(flash_apply, Q, K, V)
            bwd_flash  = bench_backward(flash_apply, Q, K, V)
            e2e_flash  = bench_fwd_bwd(flash_apply, Q, K, V)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            fwd_flash = bwd_flash = e2e_flash = float('nan')
            print(f"[Flash OOM]", end="  ")
        finally:
            torch.cuda.empty_cache()

  
        Q2, K2, V2 = make_inputs(seq_len, d, dtype, device)
        try:
            fwd_pt  = bench_forward(pytorch_apply, Q2, K2, V2)
            bwd_pt  = bench_backward(pytorch_apply, Q2, K2, V2)
            e2e_pt  = bench_fwd_bwd(pytorch_apply, Q2, K2, V2)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            fwd_pt = bwd_pt = e2e_pt = float('nan')
            print(f"[PyTorch OOM]", end="  ")
        finally:
            torch.cuda.empty_cache()

        rows.append({
            'dtype':        dtype_name,
            'seq_len':      seq_len,
            'd':            d,
            # Forward
            'fwd_flash_ms':  round(fwd_flash, 4),
            'fwd_pt_ms':     round(fwd_pt,    4),
            'fwd_speedup':   round(fwd_pt / fwd_flash, 2) if not math.isnan(fwd_flash) else float('nan'),
            # Backward
            'bwd_flash_ms':  round(bwd_flash, 4),
            'bwd_pt_ms':     round(bwd_pt,    4),
            'bwd_speedup':   round(bwd_pt / bwd_flash, 2) if not math.isnan(bwd_flash) else float('nan'),
            # End-to-end
            'e2e_flash_ms':  round(e2e_flash, 4),
            'e2e_pt_ms':     round(e2e_pt,    4),
            'e2e_speedup':   round(e2e_pt / e2e_flash, 2) if not math.isnan(e2e_flash) else float('nan'),
        })

        print(
            f"fwd  flash={fwd_flash:.3f}ms  pt={fwd_pt:.3f}ms  |  "
            f"bwd  flash={bwd_flash:.3f}ms  pt={bwd_pt:.3f}ms  |  "
            f"e2e  flash={e2e_flash:.3f}ms  pt={e2e_pt:.3f}ms"
        )

    return pd.DataFrame(rows)




def print_tables(df):
    """Print one table per (dtype, d) combo for readability."""
    for dtype_name in df['dtype'].unique():
        for d in sorted(df['d'].unique()):
            subset = df[(df['dtype'] == dtype_name) & (df['d'] == d)][
                ['seq_len',
                 'fwd_flash_ms', 'fwd_pt_ms', 'fwd_speedup',
                 'bwd_flash_ms', 'bwd_pt_ms', 'bwd_speedup',
                 'e2e_flash_ms', 'e2e_pt_ms', 'e2e_speedup']
            ].set_index('seq_len')

            print(f"\n=== dtype={dtype_name}  d={d} ===")
            print(subset.to_string())


def save_outputs(df):
    df.to_csv("flash_benchmark_results.csv", index=False)
    print("\nSaved: flash_benchmark_results.csv")


    with open("flash_benchmark_results.md", "w") as f:
        for dtype_name in df['dtype'].unique():
            for d in sorted(df['d'].unique()):
                subset = df[(df['dtype'] == dtype_name) & (df['d'] == d)][
                    ['seq_len',
                     'fwd_flash_ms', 'fwd_pt_ms', 'fwd_speedup',
                     'bwd_flash_ms', 'bwd_pt_ms', 'bwd_speedup',
                     'e2e_flash_ms', 'e2e_pt_ms', 'e2e_speedup']
                ].set_index('seq_len')
                f.write(f"\n### dtype={dtype_name}  d={d}\n\n")
                f.write(subset.to_markdown())
                f.write("\n")
    print("Saved: flash_benchmark_results.md")




if __name__ == "__main__":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}\n")

    print("Running benchmark sweep...")
    df = run_benchmark()

    print_tables(df)
    save_outputs(df)