"""
Benchmark vanilla PyTorch attention at different scales
"""

import argparse
import timeit
import math
import itertools

import torch
import pandas as pd
from a1_basics.nn_utils import softmax


BATCH_SIZE   = 8
D_MODEL_VALS = [16, 32, 64, 128]
SEQ_LEN_VALS = [256, 1024, 4096, 8192, 16384]



def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    # (batch, seq_q, seq_k)
    scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = softmax(scores, dim=-1)
    # (batch, seq_q, d_v)
    return torch.bmm(weights, V)


def make_qkv(batch, seq_len, d_model, device, requires_grad=False):
    return (
        torch.randn(batch, seq_len, d_model, device=device, requires_grad=requires_grad),
        torch.randn(batch, seq_len, d_model, device=device, requires_grad=requires_grad),
        torch.randn(batch, seq_len, d_model, device=device, requires_grad=requires_grad),
    )


def time_steps(fn, n_steps):

    times = []
    for _ in range(n_steps):
        t0 = timeit.default_timer()
        fn()
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)
    return sum(times) / len(times)


def run_benchmark(warmup, n_steps, device):
    rows = []

    for d_model, seq_len in itertools.product(D_MODEL_VALS, SEQ_LEN_VALS):
        print(f"  d_model={d_model:4d}  seq_len={seq_len:6d} ...", end="  ", flush=True)

        try:
            Q, K, V = make_qkv(BATCH_SIZE, seq_len, d_model, device, requires_grad=True)
        except torch.cuda.OutOfMemoryError:
            print("OOM (alloc)")
            rows.append(dict(d_model=d_model, seq_len=seq_len,
                             fwd_ms="OOM", mem_before_bwd_mb="OOM", bwd_ms="OOM"))
            continue

      
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    _ = scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize()

        except torch.cuda.OutOfMemoryError:
            print("OOM (warmup)")
            rows.append(dict(d_model=d_model, seq_len=seq_len,
                             fwd_ms="OOM", mem_before_bwd_mb="OOM", bwd_ms="OOM"))
            torch.cuda.empty_cache()
            continue

       
        try:
            def fwd():
                with torch.no_grad():
                    scaled_dot_product_attention(Q, K, V)

            fwd_ms = time_steps(fwd, n_steps)
            
        except torch.cuda.OutOfMemoryError:
            print("OOM (fwd timing)")
            rows.append(dict(d_model=d_model, seq_len=seq_len,
                             fwd_ms="OOM", mem_before_bwd_mb="OOM", bwd_ms="OOM"))
            torch.cuda.empty_cache()
            continue

        
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            Q2, K2, V2 = make_qkv(BATCH_SIZE, seq_len, d_model, device, requires_grad=True)
            out = scaled_dot_product_attention(Q2, K2, V2)
            torch.cuda.synchronize()

            
            mem_before_bwd_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        except torch.cuda.OutOfMemoryError:
            print("OOM (mem measure)")
            rows.append(dict(d_model=d_model, seq_len=seq_len,
                             fwd_ms=f"{fwd_ms:.2f}", mem_before_bwd_mb="OOM", bwd_ms="OOM"))
            torch.cuda.empty_cache()
            continue

        #Time backward passes 
        try:
            def bwd():
                
                Q3, K3, V3 = make_qkv(BATCH_SIZE, seq_len, d_model, device, requires_grad=True)
                result = scaled_dot_product_attention(Q3, K3, V3)
                result.sum().backward()

            bwd_ms = time_steps(bwd, n_steps)
        except torch.cuda.OutOfMemoryError:
            print("OOM (bwd)")
            rows.append(dict(d_model=d_model, seq_len=seq_len,
                             fwd_ms=f"{fwd_ms:.2f}",
                             mem_before_bwd_mb=f"{mem_before_bwd_mb:.1f}",
                             bwd_ms="OOM"))
            torch.cuda.empty_cache()
            continue

        print(f"fwd={fwd_ms:.2f}ms  mem={mem_before_bwd_mb:.1f}MB  bwd={bwd_ms:.2f}ms")
        rows.append(dict(
            d_model=d_model,
            seq_len=seq_len,
            fwd_ms=f"{fwd_ms:.2f}",
            mem_before_bwd_mb=f"{mem_before_bwd_mb:.1f}",
            bwd_ms=f"{bwd_ms:.2f}",
        ))

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup",  type=int, default=5,
                        help="Warm-up steps before timing (default: 5)")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Timed steps for each configuration (default: 100)")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\nAttention benchmark  |  batch={BATCH_SIZE}  "
          f"device={args.device}  warmup={args.warmup}  steps={args.n_steps}\n")

    rows = run_benchmark(args.warmup, args.n_steps, args.device)

    df = pd.DataFrame(rows)
    print("\n" + "="*65)
    print("Results")
    print("="*65)
    print(df.to_string(index=False))

    
    with open("attention_benchmark_results.md", "w") as f:
        f.write(df.to_markdown(index=False))
    print("\nSaved to attention_benchmark_results.md")


if __name__ == "__main__":
    main()