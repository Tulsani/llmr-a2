from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch
from a1_basics.optimizer import AdamW
import torch
import argparse
import timeit


#### definine models for sweeps ###
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

# defaults
ROPE_THETA = 10000

# parse args helper
def parser_args():
    parser = argparse.ArgumentParser()

    # select model
    parser.add_argument("--size", choices=list(MODEL_CONFIGS.keys()), default="small")

    #manual override
    parser.add_argument("--d_model",    type=int, default=None)
    parser.add_argument("--d_ff",       type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads",  type=int, default=None)

    # overrides for vocab,batch_size,rope
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--vocab_size",     type=int, default=10000)

    #benchmarking
    parser.add_argument("--mode", choices=["forward", "forward_backward", "full"],
                        default="forward")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--n_steps",      type=int, default=10)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def build_model(args):
    #model condif
    model_config = dict(MODEL_CONFIGS[args.size])

    # if over rides
    for key in ("d_model","d_ff","num_layers","num_heads"):
        val = getattr(args,key)
        if val is not None:
            model_config[key] = val
    
    #model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=ROPE_THETA,
        **model_config
    )

    # push model to device
    model.to(args.device)

    return model

# gnerate random batch
def random_batch(args):
    return torch.randint(
        0,
        args.vocab_size,
        (args.batch_size,args.context_length),
        device=args.device
    )





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



def main():
    args = parser_args()

    # build model
    print(f"\nBuilding model")
    model = build_model(args)
    # count params
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"  Parameters: {n_params:.1f}M\n")

    # genereate random data
    x = random_batch(args)

    # lets start benchmarking


if __name__ == "__main__":
    main()



        




