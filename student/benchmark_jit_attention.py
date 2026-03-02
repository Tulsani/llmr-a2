"""
 Benchmark vanilla vs torch.compile attention.
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
    scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = softmax(scores, dim=-1)
    return torch.bmm(weights, V)


# torch compile scaled
compiled_attention = torch.compile(scaled_dot_product_attention)


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


def benchmark_one(attn_fn, d_model, seq_len, warmup, n_steps, device):


    try:
        Q, K, V = make_qkv(BATCH_SIZE, seq_len, d_model, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(fwd_ms="OOM", mem_before_bwd_mb="OOM", bwd_ms="OOM")


    try:
        with torch.no_grad():
            for _ in range(warmup):
                attn_fn(Q, K, V)
                torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(fwd_ms="OOM", mem_before_bwd_mb="OOM", bwd_ms="OOM")


    try:
        def fwd():
            with torch.no_grad():
                attn_fn(Q, K, V)

        fwd_ms = time_steps(fwd, n_steps)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(fwd_ms="OOM", mem_before_bwd_mb="OOM", bwd_ms="OOM")


    try:
        torch.cuda.synchronize()
        Q2, K2, V2 = make_qkv(BATCH_SIZE, seq_len, d_model, device, requires_grad=True)
        _ = attn_fn(Q2, K2, V2)
        torch.cuda.synchronize()
        mem_before_bwd_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(fwd_ms=f"{fwd_ms:.2f}", mem_before_bwd_mb="OOM", bwd_ms="OOM")

    try:
        def bwd():
            Q3, K3, V3 = make_qkv(BATCH_SIZE, seq_len, d_model, device, requires_grad=True)
            attn_fn(Q3, K3, V3).sum().backward()

        bwd_ms = time_steps(bwd, n_steps)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(fwd_ms=f"{fwd_ms:.2f}",
                    mem_before_bwd_mb=f"{mem_before_bwd_mb:.1f}",
                    bwd_ms="OOM")

    return dict(
        fwd_ms=f"{fwd_ms:.2f}",
        mem_before_bwd_mb=f"{mem_before_bwd_mb:.1f}",
        bwd_ms=f"{bwd_ms:.2f}",
    )



def run_benchmark(warmup, n_steps, device):
    rows = []

    for d_model, seq_len in itertools.product(D_MODEL_VALS, SEQ_LEN_VALS):
        print(f"  d_model={d_model:4d}  seq_len={seq_len:6d}", flush=True)

        for label, attn_fn in [("vanilla", scaled_dot_product_attention),
                                ("compiled", compiled_attention)]:
            print(f"    [{label}] ...", end="  ", flush=True)
            result = benchmark_one(attn_fn, d_model, seq_len, warmup, n_steps, device)
            print(f"fwd={result['fwd_ms']}ms  "
                  f"mem={result['mem_before_bwd_mb']}MB  "
                  f"bwd={result['bwd_ms']}ms")

            rows.append(dict(
                d_model=d_model,
                seq_len=seq_len,
                impl=label,
                fwd_ms=result["fwd_ms"],
                mem_before_bwd_mb=result["mem_before_bwd_mb"],
                bwd_ms=result["bwd_ms"],
            ))

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup",  type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\nAttention benchmark (vanilla vs compiled)  |  "
          f"batch={BATCH_SIZE}  device={args.device}  "
          f"warmup={args.warmup}  steps={args.n_steps}\n")

    rows = run_benchmark(args.warmup, args.n_steps, args.device)

    df = pd.DataFrame(rows)

    df_pivot = df.pivot(index=["d_model", "seq_len"], columns="impl",
                        values=["fwd_ms", "mem_before_bwd_mb", "bwd_ms"])
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    print("\n" + "="*90)
    print("Results — vanilla vs compiled")
    print("="*90)
    print(df_pivot.to_string(index=False))

    with open("attention_benchmark_results.md", "w") as f:
        f.write("## Vanilla\n\n")
        f.write(df[df.impl == "vanilla"].drop(columns="impl").to_markdown(index=False))
        f.write("\n\n## Compiled\n\n")
        f.write(df[df.impl == "compiled"].drop(columns="impl").to_markdown(index=False))
        f.write("\n\n## Side-by-side\n\n")
        f.write(df_pivot.to_markdown(index=False))

    print("\nSaved to attention_benchmark_results.md")


if __name__ == "__main__":
    main()