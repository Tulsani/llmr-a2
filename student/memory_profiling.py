'''
Benchmarking script with NVTX profiling + optional BF16 mixed precision + memory profiling
'''

import argparse
import timeit
import statistics
import math
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx
import a1_basics.model
from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW
from a1_basics.nn_utils import cross_entropy, softmax
from einops import einsum

MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

ROPE_THETA = 10000


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(
            attention_weights, V,
            "... query key, ... key d_v -> ... query d_v"
        )

    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(MODEL_CONFIGS.keys()), default="small")
    parser.add_argument("--d_model",        type=int, default=None)
    parser.add_argument("--d_ff",           type=int, default=None)
    parser.add_argument("--num_layers",     type=int, default=None)
    parser.add_argument("--num_heads",      type=int, default=None)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--vocab_size",     type=int, default=10000)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "full"],
                        default="forward_backward")
    parser.add_argument("--warmup_steps",   type=int, default=5)
    parser.add_argument("--n_steps",        type=int, default=10)
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable BF16 autocast mixed precision")
    # Memory profiling — when set, skips timing and dumps a snapshot instead
    parser.add_argument("--profile_memory", action="store_true",
                        help="Dump a PyTorch memory snapshot (.pickle) instead of timing")
    parser.add_argument("--snapshot_file",  type=str, default="memory_snapshot.pickle",
                        help="Output path for memory snapshot (default: memory_snapshot.pickle)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()



def build_model(args):
    cfg = dict(MODEL_CONFIGS[args.size])
    for key in ("d_model", "d_ff", "num_layers", "num_heads"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=ROPE_THETA,
        **cfg,
    )
    model.to(args.device)
    return model


def random_batch(args):
    x = torch.randint(0, args.vocab_size,
                      (args.batch_size, args.context_length), device=args.device)
    y = torch.randint(0, args.vocab_size,
                      (args.batch_size, args.context_length), device=args.device)
    return x, y


def get_autocast_ctx(args):
    if args.mixed_precision:
        return torch.autocast(device_type=args.device, dtype=torch.bfloat16)
    return nullcontext()


def forward_backward_callable(model, x, y, args):
    optimizer = AdamW(model.parameters(), lr=3e-4)
    autocast_ctx = get_autocast_ctx(args)

    if args.mode == "forward":
        @torch.no_grad()
        def forward_pass():
            with nvtx.range("forward"):
                with autocast_ctx:
                    _ = model(x)
            torch.cuda.synchronize()
        return forward_pass

    else:  # "full"
        def full_pass():
            optimizer.zero_grad()
            with nvtx.range("forward"):
                with autocast_ctx:
                    logits = model(x)
            with nvtx.range("loss"):
                loss = cross_entropy(logits, y)
            with nvtx.range("backward"):
                loss.backward()
            with nvtx.range("optimizer"):
                optimizer.step()
            torch.cuda.synchronize()
        return full_pass


def benchmark(step_fn, warmup_steps, n_steps):
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            step_fn()

    times = []
    for i in range(n_steps):
        with nvtx.range(f"step_{i}"):
            t0 = timeit.default_timer()
            step_fn()
            t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)  # → ms

    return statistics.mean(times), statistics.stdev(times)



def profile_memory(step_fn, warmup_steps, snapshot_file):
  
    print(f"  Warming up for {warmup_steps} step(s)...", flush=True)
    for _ in range(warmup_steps):
        step_fn()

    torch.cuda.synchronize()

    print("  Starting memory recorder...", flush=True)
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # Profile exactly one step so the timeline is clean and readable
    step_fn()

    torch.cuda.memory._dump_snapshot(snapshot_file)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"  Snapshot saved to: {snapshot_file}")
    

    # Also report peak memory as a quick sanity check
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  Peak memory allocated: {peak_mb:.1f} MB")


def main():
    args = parse_args()

    a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    precision_label = "BF16 mixed" if args.mixed_precision else "FP32"
    print(f"\n  size={args.size}  ctx={args.context_length}  "
          f"mode={args.mode}  precision={precision_label}")

    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    x, y = random_batch(args)
    step_fn = forward_backward_callable(model, x, y, args)

    if args.profile_memory:
        profile_memory(step_fn, args.warmup_steps, args.snapshot_file)
    else:
        mean_ms, std_ms = benchmark(step_fn, args.warmup_steps, args.n_steps)
        print(f"  Mean : {mean_ms:8.2f} ms")
        print(f"  Std  : {std_ms:8.2f} ms\n")


if __name__ == "__main__":
    main()