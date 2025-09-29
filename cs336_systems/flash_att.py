import cs336_basics
from cs336_systems.nvx_module import annotated_scaled_dot_product_attention
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import torch
import argparse
import logging
import yaml
from timm.utils import setup_default_logging
from logging.handlers import RotatingFileHandler
import time
import statistics
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

### model
parser.add_argument('--d_model', type=int, default=512, help="model dimension")
parser.add_argument('--rope_theta', type=int, default=10000, help="RoPE theta parameter")
parser.add_argument('--n_heads', type=int, default=16, help="number of attention heads")

### benchmark options
parser.add_argument('--forward_only', action='store_true', help='only benchmark the forward pass')
parser.add_argument('--context_length', type=int, default=256, help="context length")
parser.add_argument('--batch_size', type=int, default=4, help="batch size")
parser.add_argument('--num_samples', type=int, default=10, help="number of samples to generate")

### mixedprecision
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

_logger = logging.getLogger('train')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = _parse_args()
    setup_default_logging()
    handler = RotatingFileHandler(args.log_name+'.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    att = CausalMultiHeadSelfAttention(
        d_model=args.d_model,
        num_heads=args.n_heads,
        positional_encoder=RotaryEmbedding(args.context_length, args.d_model, args.rope_theta),
    )
    _logger.info(f"att: {att}")
    random_input = torch.randn(args.num_samples, args.batch_size, args.context_length, args.d_model)
    random_input = random_input.cuda()
    att = att.cuda()
    optimizer = AdamW(att.parameters(), lr=1e-3)
    if args.mixed_precision:
        ctx_manager = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        ctx_manager = nullcontext()
    ### warmup for 5 samples
    with ctx_manager:
        with nvtx.range("warm up"):
            for i in range(5):
                if args.forward_only:
                    with torch.no_grad():
                        output = att(random_input[i])
                else:
                    output = att(random_input[i])
                    loss = output.mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
    _logger.info("warmup done")
    forward_times = []
    backward_times = []
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    with ctx_manager:
        for i in range(args.num_samples):
            if args.forward_only:
                time_start = time.perf_counter()
                with torch.no_grad():
                    output = att(random_input[i])
                torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - time_start)
            else:
                time_forward_start = time.perf_counter()
                with nvtx.range("forward"):
                    output = att(random_input[i])
                    torch.cuda.synchronize()
                    forward_times.append(time.perf_counter() - time_forward_start)
                    time_backward_start = time.perf_counter()
                with nvtx.range("backward"):
                    loss = output.mean()
                    loss.backward()
                    torch.cuda.synchronize()
                with nvtx.range("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - time_backward_start)
    torch.cuda.memory._dump_snapshot(f"memory_snapshot_forward_{args.d_model}_{args.context_length}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    # print(forward_times)
    # print(backward_times)
    if args.forward_only:
        _logger.info(f"forward avg time: {sum(forward_times) / args.num_samples} seconds")
        _logger.info(f"forward std time: {statistics.stdev(forward_times)} seconds")
    else:
        _logger.info(f"forward avg time: {sum(forward_times) / args.num_samples} seconds")
        _logger.info(f"forward std time: {statistics.stdev(forward_times)} seconds")
        _logger.info(f"backward avg time: {sum(backward_times) / args.num_samples} seconds")
        _logger.info(f"backward std time: {statistics.stdev(backward_times)} seconds")


if __name__ == "__main__":
    main()