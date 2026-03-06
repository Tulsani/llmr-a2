'''
Updating the benchmarking script to nsys profiling
'''


import argparse
import timeit
import statistics
import math

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


# annotated attention
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
    parser.add_argument("--mode", choices=["forward", "backward", "forward_backward", "full"],
                        default="forward")
    parser.add_argument("--warmup_steps",   type=int, default=5)
    parser.add_argument("--n_steps",        type=int, default=10)
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


def forward_backward_callable(model, x, y, args):
    optimizer = AdamW(model.parameters(), lr=3e-4)

    if args.mode == "forward":
        @torch.no_grad()
        def forward_pass():
            with nvtx.range("forward"):
                _ = model(x)
            torch.cuda.synchronize()

        return forward_pass

    elif args.mode == "backward":
        def backward_only():
            with nvtx.range("forward_for_graph"):
                logits = model(x)
            with nvtx.range("loss_for_graph"):
                loss = cross_entropy(logits, y)
            optimizer.zero_grad()

            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()

        return backward_only

    else:  # "forward_backward" / "full"
        def full_pass():
            optimizer.zero_grad()
            with nvtx.range("forward"):
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
        times.append((t1 - t0) * 1000)

    return statistics.mean(times), statistics.stdev(times)


def main():
    args = parse_args()

    a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    print(f"  Mode      : {args.mode}")

    x, y = random_batch(args)
    callable = forward_backward_callable(model, x, y, args)
    mean_ms, std_ms = benchmark(callable, args.warmup_steps, args.n_steps)

    print(f"\n  Mean : {mean_ms:8.2f} ms")
    print(f"  Std  : {std_ms:8.2f} ms\n")


if __name__ == "__main__":
    main()