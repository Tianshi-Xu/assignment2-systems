from torch.utils.data import Dataset
import torch
import numpy as np
import torch.distributed as dist
import math
from sys import stdout
from torch.optim import Optimizer, AdamW
from typing import Type,Any

class LMDataset(Dataset):
    def __init__(self, dataset_path: str, context_length: int,
                 input_path: str = 'input.dat', output_path: str = 'output.dat'):
        """
        dataset: 1D numpy array of token ids.
        context_length: length of context window.
        input_path/output_path: memmap filenames.
        """
        self.context_length = context_length
        self.input_path = input_path
        self.output_path = output_path
        self.dataset = np.load(dataset_path, mmap_mode="r")
        self.N = len(self.dataset) - context_length

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 返回 torch.LongTensor
        x = torch.from_numpy(self.dataset[idx:idx+self.context_length].copy()).long()
        y = torch.from_numpy(self.dataset[idx+1:idx+1+self.context_length].copy()).long()
        return x, y
    
class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1 
        def grad_hook(param: torch.Tensor):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)
            if param.requires_grad == True:
                param.register_post_accumulate_grad_hook(grad_hook)
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.mul_(1.0/self.world_size)
                

class DDP_bucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        # print("bucket_size_mb",bucket_size_mb)
        self.module = module
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1 
        self.buckets = []
        bucket = {"param":[], "total":0, "ready":0}
        bytes_per_param = next(module.parameters()).dtype.itemsize
        print("bytes_per_param", bytes_per_param)
        def hook(param: torch.Tensor):
            # print("param._bucket_idx:", param._bucket_idx)
            # print("len(self.buckets)", len(self.buckets))
            bucket = self.buckets[param._bucket_idx]
            bucket["ready"] += param.numel() * bytes_per_param
            if bucket["ready"] == bucket["total"]:
                grads = [p.grad for p in bucket["param"]]
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                bucket["handle"] = handle
                bucket["flat"] = flat_grads
                bucket["grads"] = grads
        
        for i, param in enumerate((list(module.parameters()))):
            dist.broadcast(param.data, src=0, async_op=False)
            if param.requires_grad is False:
                continue
            # print("param.data.numel()",param.data.numel())
            if bucket["total"] != 0 and bucket["total"] + param.data.numel() * bytes_per_param > bucket_size_mb * 2**20:
                self.buckets.append(bucket)
                bucket = {"param":[], "total":0, "ready":0}
            bucket["param"].append(param)
            bucket["total"] += param.data.numel() * bytes_per_param
            param._bucket_idx = len(self.buckets)
            param.register_post_accumulate_grad_hook(hook)
        if bucket["total"] > 0:
            self.buckets.append(bucket)
        print("len of buckets", len(self.buckets))
        assert len(self.buckets) > 0
        stdout.flush()
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        
    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            handle = bucket["handle"]
            handle.wait()
            bucket["flat"] *= (1/self.world_size)
            bucket["ready"] = 0
            unflat = torch._utils._unflatten_dense_tensors(bucket["flat"], bucket["grads"])
            for g_dst, g_src in zip(bucket["grads"], unflat):
               g_dst.copy_(g_src)
            bucket["handle"] = None
            bucket["flat"] = None
            bucket["grads"] = None
        # self.buckets.clear()
        
        
class FSDP_Optimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.params = list(params)
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1 
        self.local_params = []
        self.rank = dist.get_rank()
        for param in self.params:
            print(param.shape)
            rows_per_rank = param.numel() // self.world_size
            flat_param = param.view(-1)
            self.local_params.append(flat_param[rows_per_rank*self.rank:rows_per_rank*(self.rank+1)].clone().detach().requires_grad_())
            print("self.local_params[-1]:", self.local_params[-1].shape)
        self.base_optimizer = optimizer_cls(self.local_params, **kwargs)
        super().__init__(self.local_params, self.base_optimizer.defaults)
    
    def step(self, closure = None, **kwargs):
        self.base_optimizer.step(closure=closure, **kwargs)
        for param,local_param in zip(self.params,self.local_params):
            print("rank: local_param",self.rank, local_param)
            flatten_data = torch.empty((param.data.numel()), device=local_param.device)
            dist.all_gather_into_tensor(flatten_data, local_param.data, async_op=False)
            print("rank: flatten_data",self.rank, flatten_data)
            param.data = flatten_data.view(param.data.shape)
        
    def add_param_group(self, param_group: dict[str, Any]):
        for group in param_group["params"]:
            param = group
            rows_per_rank = param.numel() // self.world_size
            flat_param = param.view(-1)
            self.local_params.append(flat_param[rows_per_rank*self.rank:rows_per_rank*(self.rank+1)].clone().detach().requires_grad_())