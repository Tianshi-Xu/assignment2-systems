import torch
import torch.nn as nn
import torch.distributed as dist
import torch
import os
import torch.multiprocessing as mp
from utils import *
from torch.utils.data import DataLoader
from cs336_basics import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import time
import torch.cuda.nvtx as nvtx
from torch.nn.parallel import DistributedDataParallel as torch_DDP

class toy_network(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        layers = []
        for i in range(3):
            layers.append(nn.Linear(128, 128))
            layers.append(nn.ReLU())
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)
    
def setup(backend, rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def train_worker(rank, world_size, train_loader, vaild_loader, epoch, batch_size):
    setup("nccl", rank, world_size)
    device = f"cuda:{rank+2}"
    num_tokens = 32768000
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000,
    ).to(device)
    ### broadcast param to all GPUs
    for param in model.parameters():
        dist.broadcast(param.data, src=0, async_op=False)
    print("check param equal:", torch.mean(next(model.parameters())))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    local_batch_size = batch_size // world_size
    print("local batch size:", local_batch_size)
    best_loss = 1e6
    n_steps = num_tokens//(batch_size*256)
    n_steps = 10
    for i in range(epoch):
        for j, (input, target) in enumerate(train_loader):
            if j > n_steps:
                break
            torch.cuda.synchronize()
            with nvtx.range("fwd and bwd"):
                total_time_start = time.perf_counter()
                local_input = input[rank*local_batch_size:(rank+1)*local_batch_size:].to(device)
                local_target = target[rank*local_batch_size:(rank+1)*local_batch_size:].to(device)
                y = model(local_input)
                loss = cross_entropy(y, local_target)
                if best_loss > loss:
                    best_loss = loss
                loss.backward()
                # torch.cuda.synchronize()
            if rank == 0 and j > 5:
                print(f"fwd and bwd time: {time.perf_counter()-total_time_start}")
            with nvtx.range("all reduce"):
                all_reduce_time_start = time.perf_counter()
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG,async_op=False)
                # dist.barrier()
                # torch.cuda.synchronize()
            if rank == 0 and j > 5:
                print(f"all reduce time: {time.perf_counter()-all_reduce_time_start}")
            with nvtx.range("optimizer"):
                optimizer.step()
                optimizer.zero_grad()
                if rank == 0 and j % 10 ==0:
                    print(f"step:{j}, best_loss:{best_loss}")
                torch.cuda.synchronize()
            if rank == 0 and j > 5:
                print(f"total time: {time.perf_counter()-total_time_start}")
    test_param = next(model.parameters())
    dist.destroy_process_group()

def train_worker_fast(rank, world_size, train_loader, vaild_loader, epoch, batch_size):
    setup("nccl", rank, world_size)
    device = f"cuda:{rank+2}"
    num_tokens = 32768000
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000,
    ).to(torch.bfloat16).to(device)
    # model = DDP(model)
    model = DDP_bucketed(model, 1)
    # model = torch_DDP(model, device_ids=[rank+2])
    print("check param equal:", torch.mean(next(model.parameters())))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    local_batch_size = batch_size // world_size
    print("local batch size:", local_batch_size)
    best_loss = 1e6
    n_steps = num_tokens//(batch_size*256)
    n_steps = 10
    for i in range(epoch):
        for j, (input, target) in enumerate(train_loader):
            if j > n_steps:
                break
            torch.cuda.synchronize()
            with nvtx.range("fwd and bwd"):
                total_time_start = time.perf_counter()
                local_input = input[rank*local_batch_size:(rank+1)*local_batch_size:].to(device)
                local_target = target[rank*local_batch_size:(rank+1)*local_batch_size:].to(device)
                y = model(local_input)
                loss = cross_entropy(y, local_target)
                if best_loss > loss:
                    best_loss = loss
                loss.backward()
                # torch.cuda.synchronize()
            if rank == 0 and j > 5:
                print(f"fwd and bwd time: {time.perf_counter()-total_time_start}")
                # torch.cuda.synchronize()
            model.finish_gradient_synchronization()
            with nvtx.range("optimizer"):
                optimizer.step()
                optimizer.zero_grad()
                if rank == 0 and j % 10 ==0:
                    print(f"step:{j}, best_loss:{best_loss}")
                torch.cuda.synchronize()
            if rank == 0 and j > 5:
                print(f"total time: {time.perf_counter()-total_time_start}")
    test_param = next(model.parameters())
    print("check param equal:", torch.mean(test_param))
    stdout.flush()
    dist.destroy_process_group()
   
 
def main():
    world_size = 2
    epoch = 1
    batch_size = 32
    train_dir = "data/TinyStories_train_tokens.npy"
    valid_dir = "data/TinyStories_valid_tokens.npy"
    context_length = 128
    train_dataset = LMDataset(train_dir, context_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = LMDataset(valid_dir,context_length)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    mp.spawn(train_worker_fast,args=(world_size,train_loader, valid_loader, epoch, batch_size), nprocs=world_size, join=True,)
    
if __name__ == "__main__":
    main()