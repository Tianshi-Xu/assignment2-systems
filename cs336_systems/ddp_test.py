import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from sys import stdout

def setup(backend, rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, warmup, backend, data_size, world_size):
    setup(backend, rank, world_size)
    if backend == "gloo":
        device = "cpu"
    else:
        device = f"cuda:{rank}"
    data = torch.randint(0, 10, (data_size,), dtype=torch.float32, device=device)
    for i in range(5):
        dist.all_reduce(data, async_op=False)
    dist.barrier()
    if backend == "nccl":
        torch.cuda.synchronize()
    time_start= time.perf_counter()
    dist.all_reduce(data, async_op=False)
    dist.barrier()
    if backend == "nccl":
        torch.cuda.synchronize()
    latency= time.perf_counter() - time_start
    all_times = [None for _ in range(world_size)]
    if backend == "gloo":
        dist.all_gather_object(all_times, latency)
    else:
        latency = torch.tensor([latency], device=device)
        all_times = [torch.zeros_like(latency) for _ in range(world_size)]
        dist.all_gather(all_times, latency)
    if rank == 0 and not warmup:
        if backend == "nccl":
            all_times = [t.item() for t in all_times]
        print(f"rank 0 backend:{backend}, data_size:{data_size}, world_size:{world_size}, time:{sum(all_times)/world_size}")
        stdout.flush()
    dist.destroy_process_group()
    
def bench_all_reduce(backend, data_size, world_size):
    num_step = 1
    time_list = []
    for i in range(num_step):
        time_start = time.perf_counter()
        mp.spawn(fn=distributed_demo, args=(False, backend, data_size, world_size, ), nprocs=world_size, join=True,)
        if backend == "nccl":
            torch.cuda.synchronize()
        time_list.append(time.perf_counter()-time_start)
    # print(f"backend:{backend}, data_size:{data_size}, world_size:{world_size}, time:{sum(time_list)/num_step}")
    # stdout.flush()
    
if __name__ == "__main__":
    world_sizes = [2,3,4]
    # world_sizes = [2]
    backends = ["gloo","nccl"]
    data_sizes = [2**18, 10*2**18, 100*2**18, 2**28, 2**29]
    # data_sizes = [10*2**28]
    for backend in backends:
        for world_size in world_sizes:
            for data_size in data_sizes:
                bench_all_reduce(backend, data_size, world_size)