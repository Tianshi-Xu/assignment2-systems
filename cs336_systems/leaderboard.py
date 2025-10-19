
import torch
import math
from einops import rearrange,einsum,reduce
import triton
import triton.language as tl
from flash_att import FlashAttentionTriton
from torch.nn.functional import scaled_dot_product_attention

def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3,1, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(FlashAttentionTriton.apply)
    # flash = scaled_dot_product_attention

    def flash_forward_backward():
        # print("q.shape",q.shape)
        o = flash(q, k, v, False)
        # loss = o.sum()
        # loss.backward()
    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)
    
if __name__ == "__main__":
    test_timing_flash_forward_backward()