from a1_basics.model import BasicsTransformerLM
from a1_basics.data import get_batch
from a1_basics.optimizer import AdamW
import torch
import argparse
import timeit
from a1_basics.nn_utils import cross_entropy


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
    parser.add_argument("--mode", choices=["forward", "forward_backward", "backward"],
                        default="forward")
    parser.add_argument("--warmup_steps", type=int)
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
    x = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size,args.context_length),
        device=args.device
    )
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    return x,y



# getting this from examples
def mean_fn(values: list[float]) -> float:
    return sum(values) / len(values)


def forward_backward_callable(model, x, y, args):

    if args.mode == "forward":
        @torch.no_grad()
        def forward_pass():
            logits = model(x)
            torch.cuda.synchronize()
        return forward_pass

    if args.mode == "backward":
        # Run forward once outside timing to get a live computation graph
        logits = model(x)
        loss = cross_entropy(logits, y)

        def backward_pass():
            # Retain graph so we can call backward repeatedly across steps
            loss.backward(retain_graph=True)
            torch.cuda.synchronize()

        return backward_pass

    if args.mode == "forward_backward":
        optimizer = AdamW(model.parameters(), lr=3e-4)

        def full_pass():
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

        return full_pass
    
    #like example
    if args.mode == "forward":
        
        @torch.no_grad()
        def forward_pass():
            logits = model(x)
            torch.cuda.synchronize()

        return forward_pass
    
    if args.mode == "forward_backward":

        optimzer = AdamW(model.parameters(),lr=3e-4)

        def full_pass():
            #zero all grads
            optimzer.zero_grad()

            # forward pass
            logits = model(x)
            # loss
            loss = cross_entropy(logits,y)
            #
            loss.backward()
            # optimzer step
            optimzer.step()

            torch.cuda.synchronize()

        return full_pass

def benchmark(callable,warmup_steps=None,n_steps=None):
    
    if warmup_steps is not None: 
        print("Warm up steps being used")
        for _ in range(warmup_steps):
            callable()

    times = []

    for _ in range(n_steps):
        start_time = timeit.default_timer()

        # call
        callable()

        # end time
        end_time = timeit.default_timer()

        #times
        times.append((end_time-start_time))

    mean = mean_fn(times)

    return mean



def main():
    args = parser_args()

    # build model
    print(f"\nBuilding model")
    model = build_model(args)
    # count params
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"  Parameters: {n_params:.1f}M\n")

    # genereate random data
    x,y = random_batch(args)

    # get callable
    callable = forward_backward_callable(model,x,y,args)

    # lets start benchmarking
    mean_s = benchmark(callable,args.warmup_steps,args.n_steps)
    mean_ms = mean_s * 1000.0

    # need to return this as table
    print(f"\nResults  [{args.mode} | {args.size} | ctx={args.context_length}]")
    print(f"  Mean : {mean_ms:8.2f} ms")




if __name__ == "__main__":
    main()



        




