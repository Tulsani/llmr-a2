from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch
from a1_basics.optimizer import AdamW
import torch
import time
import timeit
import torch

# getting this from examples
def mean(values: list[float]) -> float:
    return sum(values) / len(values)

@torch.no_grad()
def _forward_only(model, x):
    return model(x)

def benchmark(
    model,
    x,
    y,
    optimizer,
    loss_fn,
    warmup_steps=10,
    steps=50,
    mode="forward",  # "forward" or "forward_backward"
):
    device = next(model.parameters()).device
    model.train()

    for _ in range(warmup_steps):
        if mode == "forward":
            _ = _forward_only(model, x)
        elif mode == "forward_backward":
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if device.type == "cuda":
            torch.cuda.synchronize()  # per spec: after each step

    times = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()  # flush before timing

        t0 = timeit.default_timer()

        if mode == "forward":
            _ = _forward_only(model, x)
        else:  # forward_backward
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize() 

        t1 = timeit.default_timer()
        times.append(t1 - t0)

    return mean(times), times



        
        




