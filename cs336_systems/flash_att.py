from typing import Any
import torch
import math
from einops import rearrange,einsum,reduce
import triton
import triton.language as tl
import time
from jaxtyping import Float, Bool
from torch import Tensor
from sys import stdout
import torch.cuda.nvtx as nvtx
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.llama.modeling_llama import LlamaAttention
from torch.nn.attention import sdpa_kernel
from torch.backends.cuda import SDPBackend  
from triton_tutorial import _attention

class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q = 16
        B_k = 16
        batch, sqlen, d = Q.shape
        T_q = sqlen // B_q
        T_k = K.shape[1] // B_k
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((batch, sqlen), device=Q.device) ## (batch, sqlen)
        for i in range(T_q):
            Q_i = Q[:,i*B_q:(i+1)*B_q,:]
            O_ij = torch.zeros_like(Q_i, device=Q.device)
            l_ij = torch.zeros((batch, B_q, 1), device=Q.device)
            m_ij = torch.full((batch, B_q, 1), -torch.inf, device=Q.device)    ## negetive infinity of shape b_q
            new_m_ij = torch.empty_like(m_ij, device=Q.device)
            for j in range(T_k):
                K_j = K[:,j*B_k:(j+1)*B_k,:]
                V_j = V[:,j*B_k:(j+1)*B_k,:]
                S_ij = einsum(Q_i, K_j, "batch B_q d, batch B_k d -> batch B_q B_k")/math.sqrt(d)
                if is_causal:
                    offset_q = torch.arange(B_q)[:,None]
                    offset_k = torch.arange(B_k)[None,:]
                    mask = (offset_q + i*B_q) < (offset_k + j*B_k)
                    mask = mask.to(S_ij.device)
                    S_ij = S_ij + mask * (-1e6)
                # print("S_i:", S_ij)
                new_m_ij = torch.max(m_ij, reduce(S_ij, "batch B_q B_k -> batch B_q 1", "max"))
                P_ij = torch.exp(S_ij-new_m_ij)
                tmp = torch.exp(m_ij-new_m_ij)
                l_ij = tmp * l_ij + reduce(P_ij, "batch B_q B_k -> batch B_q 1", "sum")
                O_ij = tmp * O_ij + einsum(P_ij, V_j.float(), "batch B_q B_k, batch B_k d -> batch B_q d")
                m_ij = new_m_ij
            O[:,i*B_q:(i+1)*B_q,:] = O_ij / l_ij
            L[:,i*B_q:(i+1)*B_q] = (m_ij + torch.log(l_ij)).squeeze(-1)
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        # print("O.shape", O.shape)
        # print("L.shape", L.shape)
        return O.float()

    @staticmethod
    def backward(ctx, grad_out):
        # print("grad_out.dtype:", grad_out.dtype)
        Q,K,V,L,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        d = Q.shape[-1]
        S = einsum(Q,K,"batch N_q d, batch N_k d -> batch N_q N_k")/math.sqrt(d)
        # print("S:", S)
        if is_causal:
            mask = torch.arange(S.shape[-2])[None,:] > torch.arange(S.shape[-1])[:,None]
            mask = mask.to(S.device)
            mask = torch.where(mask, -1e6, 0)
            S += mask
        P = torch.exp(S-L.unsqueeze(-1))
        grad_V = einsum(P,grad_out,"batch N_q N_k, batch N_q d -> batch N_k d")
        
        grad_P = einsum(grad_out, V.float(), "batch N_q d, batch N_k d -> batch N_q N_k")
        # print("grad_V:", grad_V)
        D = reduce(O*grad_out, "batch N_q d -> batch N_q 1", "sum")
        grad_S = P * (grad_P - D)
        grad_Q = einsum(grad_S,K.float(),"batch N_q N_k, batch N_k d -> batch N_q d")/math.sqrt(d)
        # print("grad_Q:", grad_Q)
        grad_K = einsum(grad_S,Q.float(),"batch N_q N_k, batch N_q d -> batch N_k d")/math.sqrt(d)
        return grad_Q, grad_K, grad_V, None

def naive_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    is_causal: Bool,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if is_causal:
        mask = torch.arange(Q.shape[-2])[:,None] < torch.arange(K.shape[-2])[None, :]
        attention_scores = attention_scores + (mask * (-1e6)).to(attention_scores.device)

    attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V.float(), "... query key, ... key d_v ->  ... query d_v")


@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16, "K_TILE_SIZE": 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32, "K_TILE_SIZE": 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, "K_TILE_SIZE": 64}, num_warps=4, num_stages=3),
        triton.Config({'Q_TILE_SIZE': 64, "K_TILE_SIZE": 64}, num_warps=4, num_stages=4),
        triton.Config({'Q_TILE_SIZE': 128, "K_TILE_SIZE": 128}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16, "K_TILE_SIZE": 16}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32, "K_TILE_SIZE": 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64, "K_TILE_SIZE": 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, "K_TILE_SIZE": 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, "K_TILE_SIZE": 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, "K_TILE_SIZE": 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 256, "K_TILE_SIZE": 256}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, "K_TILE_SIZE": 64}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 128, "K_TILE_SIZE": 128}, num_warps=16),
    ],
    key=['D', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
)
@triton.heuristics(
    {
        'Q_TILE_SIZE': lambda args: min(args['Q_TILE_SIZE'], args['N_QUERIES']),
    }
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
):
    start_pid = tl.program_id(0)
    
    num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    num_total_tiles = num_q_tiles * NUM_BATCHES
    # 创建当前 tile 的 block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(start_pid * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(start_pid * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(start_pid * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    K_desc = tl.make_tensor_descriptor(
        K_ptr,
        shape=(NUM_BATCHES, N_KEYS, D),
        strides=(stride_kb, stride_kk, stride_kd),
        # offsets=(0, 0),
        block_shape=(1, K_TILE_SIZE, D),
        # order=(1, 0),
    )
    V_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=(NUM_BATCHES, N_KEYS, D),
        strides=(stride_vb, stride_vk, stride_vd),
        # offsets=(0, 0),
        block_shape=(1, K_TILE_SIZE, D),
        # order=(1, 0),
    )
    # 关键修改：循环遍历所有 tiles，不仅仅是一个 batch 的
    for tile_id in tl.range(start_pid, num_total_tiles, NUM_SMS):
        # 从 tile_id 计算出 batch_index 和 query_tile_index
        batch_index = tile_id // num_q_tiles
        query_tile_index = tile_id % num_q_tiles
        
        # 加载 Q 并进行计算
        Q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
        m_i = tl.full((Q_TILE_SIZE, 1), -float("inf"), dtype=tl.float32)
        new_m_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
        if is_causal:
            offset_q = tl.arange(0, Q_TILE_SIZE)
            offset_q = tl.expand_dims(offset_q, 1)
            offset_k = tl.arange(0, K_TILE_SIZE)
            offset_k = tl.expand_dims(offset_k, 0)
        for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            K_i = K_desc.load([batch_index, i * K_TILE_SIZE, 0]).reshape(K_TILE_SIZE,D)
            V_i = V_desc.load([batch_index, i * K_TILE_SIZE, 0]).reshape(K_TILE_SIZE,D)
            S_i = tl.dot(Q, tl.trans(K_i), out_dtype=tl.float32)
            S_i = S_i * scale
            if is_causal:
                mask = (offset_q + query_tile_index * Q_TILE_SIZE) >= (offset_k + i * K_TILE_SIZE)
                S_i = tl.where(mask, S_i, -float("inf"))
            new_m_i = tl.maximum(m_i, tl.max(S_i, axis=1, keep_dims=True))
            P_i = tl.exp(S_i - new_m_i)
            tmp = tl.exp(m_i - new_m_i)
            l_i = tmp * l_i + tl.sum(P_i, axis=1, keep_dims=True)
            O_i = tmp * O_i + tl.dot(P_i.to(INPUT_DTYPE), V_i, out_dtype=tl.float32)
            m_i = new_m_i
        
        O_i = O_i / l_i
        tl.store(O_block_ptr, O_i.to(INPUT_DTYPE), boundary_check=(0,1))
        tl.store(L_block_ptr, (m_i + tl.log(l_i)).reshape(Q_TILE_SIZE), boundary_check=(0,))
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE*NUM_SMS, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE*NUM_SMS, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE*NUM_SMS, ))

@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=8, num_stages=3),
        triton.Config({'Q_TILE_SIZE': 256}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 256}, num_warps=16),
    ],
    key=['d', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
    reset_to_zero=['dQ_ptr'], 
)

@triton.heuristics(
    {
        'Q_TILE_SIZE': lambda args: min(args['Q_TILE_SIZE'], args['N_QUERIES']),
        'K_TILE_SIZE': lambda args: min(args['Q_TILE_SIZE'], args['N_KEYS']),
    }
)

@triton.jit
def flash_bwd_kernel_slow(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_ll,
    stride_Db, stride_DD,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    # Program indices
    kv_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vv, stride_vd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_ll,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oo, stride_od),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(kv_tile_index * K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1,0)
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vv, stride_vd),
        offsets=(kv_tile_index * K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1,0)
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES, ),
        strides=(stride_DD, ),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )
    K = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
    V = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
    dK = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        S_i = tl.dot(Q_i, tl.trans(K), out_dtype=tl.float32) * scale
        # tl.device_print("S_i",S_i)
        if is_causal:
            offset_q = tl.arange(0,Q_TILE_SIZE)
            offset_q = tl.expand_dims(offset_q,1)
            offset_k = tl.arange(0,K_TILE_SIZE)
            offset_k = tl.expand_dims(offset_k,0)
            mask = (offset_q + i * Q_TILE_SIZE) >= (offset_k + kv_tile_index * K_TILE_SIZE)
            S_i = tl.where(mask, S_i, -float("inf"))
            # tl.device_print("mask", mask)
        L_i = tl.expand_dims(L_i,1)
        # tl.device_print("L_i",L_i)
        # tl.device_print("S_i-L_i",S_i-L_i)
        P_i = tl.exp(S_i-L_i)
        # tl.device_print("P_i",P_i)
        # tl.device_print("dO_i",dO_i)
        dV += tl.dot(tl.trans(P_i), dO_i, out_dtype=tl.float32)
        # tl.device_print("dV",dV)
        dP_i = tl.dot(dO_i.to(INPUT_DTYPE), tl.trans(V), out_dtype=tl.float32)
        dS_i = P_i * (dP_i - tl.expand_dims(D_i,1)) * scale
        tmp = tl.dot(dS_i.to(INPUT_DTYPE), K, out_dtype=tl.float32)
        offs_q = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   # shape = [Q_TILE_SIZE]
        # hidden 维度
        offs_d = tl.arange(0, d) 
        ptrs = (
            dQ_ptr
            + batch_index * stride_qb
            + offs_q[:, None] * stride_qq     # query offset
            + offs_d[None, :] * stride_qd     # hidden offset
        )
        # tl.device_print("offsets",batch_index * stride_qb
        #     + offs_q[:, None] * stride_qq
        #     + offs_d[None, :] * stride_qd)
        tl.atomic_add(ptrs, tmp)
        # tl.device_print("dQ_i",dQ_i)
        # tl.device_print("tmp",tmp)
        dK += tl.dot(tl.trans(dS_i).to(INPUT_DTYPE),Q_i, out_dtype=tl.float32)
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, ))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
    tl.store(dK_block_ptr, dK)
    tl.store(dV_block_ptr, dV)

@triton.jit
def flash_bwd_kernel_fast_dKV(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_ll,
    stride_Db, stride_DD,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
        # Program indices
    kv_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vv, stride_vd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_ll,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oo, stride_od),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(kv_tile_index * K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1,0)
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vv, stride_vd),
        offsets=(kv_tile_index * K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1,0)
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES, ),
        strides=(stride_DD, ),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )
    K = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
    V = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
    dK = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        S_i = tl.dot(Q_i, tl.trans(K), out_dtype=tl.float32) * scale
        # tl.device_print("S_i",S_i)
        if is_causal:
            offset_q = tl.arange(0,Q_TILE_SIZE)
            offset_q = tl.expand_dims(offset_q,1)
            offset_k = tl.arange(0,K_TILE_SIZE)
            offset_k = tl.expand_dims(offset_k,0)
            mask = (offset_q + i * Q_TILE_SIZE) >= (offset_k + kv_tile_index * K_TILE_SIZE)
            S_i = tl.where(mask, S_i, -float("inf"))
            # tl.device_print("mask", mask)
        L_i = tl.expand_dims(L_i,1)
        # tl.device_print("L_i",L_i)
        # tl.device_print("S_i-L_i",S_i-L_i)
        P_i = tl.exp(S_i-L_i)
        # tl.device_print("P_i",P_i)
        # tl.device_print("dO_i",dO_i)
        dV += tl.dot(tl.trans(P_i), dO_i, out_dtype=tl.float32)
        # tl.device_print("dV",dV)
        dP_i = tl.dot(dO_i.to(INPUT_DTYPE), tl.trans(V), out_dtype=tl.float32)
        dS_i = P_i * (dP_i - tl.expand_dims(D_i,1)) * scale
        dK += tl.dot(tl.trans(dS_i).to(INPUT_DTYPE),Q_i, out_dtype=tl.float32)
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, ))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
    tl.store(dK_block_ptr, dK)
    tl.store(dV_block_ptr, dV)


@triton.jit
def flash_bwd_kernel_fast_dQ(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dO_ptr,dQ_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_ll,
    stride_Db, stride_DD,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vv, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0)
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_ll,),
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oo, stride_od),
        offsets=(q_tile_index * Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES, ),
        strides=(stride_DD, ),
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )
    Q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    D = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    D = tl.expand_dims(D,1)
    dO = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero").to(INPUT_DTYPE)
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    L = tl.expand_dims(L,1)
    dQ = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_i = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_i = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        S_i = tl.dot(Q, tl.trans(K_i), out_dtype=tl.float32) * scale
        # tl.device_print("S_i",S_i)
        if is_causal:
            offset_q = tl.arange(0,Q_TILE_SIZE)
            offset_q = tl.expand_dims(offset_q,1)
            offset_k = tl.arange(0,K_TILE_SIZE)
            offset_k = tl.expand_dims(offset_k,0)
            mask = (offset_q + q_tile_index * Q_TILE_SIZE) >= (offset_k + i * K_TILE_SIZE)
            S_i = tl.where(mask, S_i, -float("inf"))
            # tl.device_print("mask", mask)
        # tl.device_print("L_i",L_i)
        # tl.device_print("S_i-L_i",S_i-L_i)
        P_i = tl.exp(S_i-L)
        # tl.device_print("P_i",P_i)
        # tl.device_print("dO_i",dO_i)
        # tl.device_print("dV",dV)
        dP_i = tl.dot(dO, tl.trans(V_i), out_dtype=tl.float32)
        dS_i = P_i * (dP_i - D) * scale
        dQ += tl.dot(dS_i.to(INPUT_DTYPE), K_i, out_dtype=tl.float32)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ)


@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 256, 'K_TILE_SIZE': 256}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=16),
    ],
    key=['d', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
)

@triton.heuristics(
    {
        'Q_TILE_SIZE': lambda args: min(args['Q_TILE_SIZE'], args['N_QUERIES']),
        'K_TILE_SIZE': lambda args: min(args['K_TILE_SIZE'], args['N_KEYS']),
    }
)
@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_ll,
    stride_Db, stride_DD,
    N_BATCHES: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    grid_dkv = lambda meta: (
        triton.cdiv(N_KEYS, K_TILE_SIZE),
        N_BATCHES,
    )
    grid_dq = lambda meta: (
        triton.cdiv(N_QUERIES, Q_TILE_SIZE),
        N_BATCHES,
    )
    flash_bwd_kernel_fast_dKV[grid_dkv](
        Q_ptr, K_ptr, V_ptr,
        L_ptr, D_ptr,
        dO_ptr, dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vv, stride_vd,
        stride_ob, stride_oo, stride_od,
        stride_lb, stride_ll,
        stride_Db, stride_DD,
        N_QUERIES, N_KEYS,
        scale,
        d,
        is_causal,
        INPUT_DTYPE,
    )
    flash_bwd_kernel_fast_dQ[grid_dq](
        Q_ptr, K_ptr, V_ptr,
        L_ptr, D_ptr,
        dO_ptr, dQ_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vv, stride_vd,
        stride_ob, stride_oo, stride_od,
        stride_lb, stride_ll,
        stride_Db, stride_DD,
        N_QUERIES, N_KEYS,
        scale,
        d,
        is_causal,
        INPUT_DTYPE,
    )
    return

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if len(Q.shape) == 3:
            Q = Q.unsqueeze(1)
        # print(Q.shape)
        batch, num_head, num_q, d = Q.shape
        num_k = K.shape[2]
        dtype_map = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }
        Q=Q.view(-1, num_q, d)
        K=K.view(-1, num_k, d)
        V=V.view(-1, num_k, d)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 4
        grid = lambda meta: (
            min(NUM_SMS, triton.cdiv(num_q, meta['Q_TILE_SIZE']) * batch * num_head),
        )
        input_dtype_triton = dtype_map.get(Q.dtype, tl.float32)
        O = torch.empty_like(Q, device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch * num_head, num_q), device=Q.device, dtype=torch.float32)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=num_q, N_KEYS=num_k,
            scale=1/math.sqrt(d),
            D=d,
            is_causal=is_causal,
            INPUT_DTYPE=input_dtype_triton,
            NUM_SMS=NUM_SMS,
            NUM_BATCHES=batch * num_head,
            ### autotune, Q_TILE_SIZE and K_TILE_SIZE will be replaced by the best config
        )
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.num_head = num_head
        return O.view(batch, num_head, num_q, d)
    
    @staticmethod
    @torch.compile
    def backward_slow(ctx, grad_out):
        Q,K,V,L,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        d = Q.shape[-1]
        S = einsum(Q,K,"batch N_q d, batch N_k d -> batch N_q N_k")/math.sqrt(d)
        if is_causal:
            mask = torch.arange(S.shape[1])[None,:] > torch.arange(S.shape[1])[:,None]
            mask = mask.to(S.device)
            S = S + mask * (-1e6)
        P = torch.exp(S-L.unsqueeze(-1))
        grad_V = einsum(P,grad_out,"batch N_q N_k, batch N_q d -> batch N_k d")
        grad_P = einsum(grad_out, V, "batch N_q d, batch N_k d -> batch N_q N_k")
        D = reduce(O*grad_out, "batch N_q d -> batch N_q 1", "sum")
        grad_S = P * (grad_P - D)
        grad_Q = einsum(grad_S,K,"batch N_q N_k, batch N_k d -> batch N_q d")/math.sqrt(d)
        grad_K = einsum(grad_S,Q,"batch N_q N_k, batch N_q d -> batch N_k d")/math.sqrt(d)
        return grad_Q, grad_K, grad_V, None
    
    @staticmethod
    def backward_normal(ctx: Any, grad_out: Any) -> Any:
        grad_out = grad_out.to(torch.float32)
        Q,K,V,L,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        grid = lambda meta: (
            triton.cdiv(n_keys, meta['K_TILE_SIZE']),
            batch,
        )
        dtype_map = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }
        input_dtype_triton = dtype_map.get(Q.dtype, tl.float32)
        D = reduce(O.float() * grad_out, "batch N_q d -> batch N_q", "sum")
        dQ = torch.zeros_like(Q, device=Q.device, dtype=torch.float32)
        dK = torch.empty_like(K, device=K.device, dtype=torch.float32)
        dV = torch.empty_like(V, device=V.device, dtype=torch.float32)
        flash_bwd_kernel[grid](
            Q,K,V,
            L,D,
            grad_out,dQ,dK,dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            L.stride(0),L.stride(1),
            D.stride(0),D.stride(1),
            n_queries, n_keys,
            scale=1/math.sqrt(d),
            d=d,
            is_causal=is_causal,
            INPUT_DTYPE=input_dtype_triton,
        )
        return dQ,dK,dV,None
    
    @staticmethod
    def backward(ctx: Any, grad_out: Any) -> Any:
        grad_out = grad_out.to(torch.float32)
        Q,K,V,L,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch, n_head, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        dtype_map = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }
        grid = lambda meta: (
            triton.cdiv(n_keys, meta['K_TILE_SIZE']),
            batch,
        )
        grid_dq = lambda meta: (
            triton.cdiv(n_queries, meta['Q_TILE_SIZE']),
            batch,
        )
        input_dtype_triton = dtype_map.get(Q.dtype, tl.float32)
        D = reduce(O.float() * grad_out, "... N_q d -> ... N_q", "sum")
        dQ = torch.empty_like(Q, device=Q.device, dtype=torch.float32)
        dK = torch.empty_like(K, device=K.device, dtype=torch.float32)
        dV = torch.empty_like(V, device=V.device, dtype=torch.float32)
        flash_bwd_kernel
        dQ = dQ.view(-1, ctx.num_head, n_queries, d)
        dK = dK.view(-1, ctx.num_head, n_keys, d)
        dV = dV.view(-1, ctx.num_head, n_keys, d)
        return dQ,dK,dV,None

def test_triton():
    ground_truth = scaled_dot_product_attention
    pytorch_func = FlashAttentionTorch.apply
    triton_func = FlashAttentionTriton.apply
    input_type = torch.bfloat16
    Q = torch.randn(1,1,32768,64, requires_grad=True,device="cuda", dtype=input_type)
    K = torch.randn(1,1,32768,64, requires_grad=True,device="cuda", dtype=input_type)
    V = torch.randn(1,1,32768,64, requires_grad=True,device="cuda", dtype=input_type)
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        O_pytorch = ground_truth(Q, K, V, is_causal=False)
    dO = 0.1*torch.randn_like(O_pytorch, dtype=torch.float32)
    print("O_pytorch.dtype:", O_pytorch.dtype)
    print("dO.dtype:", dO.dtype)
    O_pytorch.backward(dO)
    Q_grad_pytorch = Q.grad.clone()
    K_grad_pytorch = K.grad.clone()
    V_grad_pytorch = V.grad.clone()
    # print("V_grad_pytorch:",V_grad_pytorch)
    input_type = torch.bfloat16
    Q.grad.zero_()
    K.grad.zero_()
    V.grad.zero_()
    Q = Q.detach().to(input_type).requires_grad_(True)
    K = K.detach().to(input_type).requires_grad_(True)
    V = V.detach().to(input_type).requires_grad_(True)
    O_triton = triton_func(Q, K, V, False)
    O_triton.backward(dO)
    Q_grad_triton = Q.grad.clone()
    K_grad_triton = K.grad.clone()
    V_grad_triton = V.grad.clone()
    # print(flash_bwd_kernel.best_config)
    # print("V_grad_triton:",V_grad_triton)
    print(O_pytorch.flatten()[-10:])
    print(O_triton.flatten()[-10:])
    print(torch.sum((O_pytorch-O_triton).abs() <= 1e-3))
    print("output max error:", torch.max(torch.abs(O_pytorch - O_triton)))
    print("Q grad max error:", torch.max(torch.abs(Q_grad_pytorch - Q_grad_triton)))
    print("K grad max error:", torch.max(torch.abs(K_grad_pytorch - K_grad_triton)))
    print("V grad max error:", torch.max(torch.abs(V_grad_pytorch - V_grad_triton)))
    print("output max error:", torch.allclose(O_pytorch, O_triton, atol=1e-2, rtol=1e-3))
    print("Q grad max error:", torch.allclose(Q_grad_pytorch, Q_grad_triton, atol=1e-2, rtol=1e-3))
    print("K grad max error:", torch.allclose(K_grad_pytorch, K_grad_triton, atol=1e-2, rtol=1e-3))
    print("V grad max error:", torch.allclose(V_grad_pytorch, V_grad_triton, atol=1e-2, rtol=1e-3))
    print(flash_fwd_kernel.best_config)

def triton_bench(use_triton, sqlen, head_dim,num_head, mode, samples=5, warmup_steps=5):
    torch.set_float32_matmul_precision('high')
    input_type = torch.bfloat16
    Q  = torch.randn(1,num_head,sqlen,head_dim,requires_grad=True, device="cuda", dtype=input_type)
    K  = torch.randn(1,num_head,sqlen,head_dim,requires_grad=True, device="cuda", dtype=input_type)
    V  = torch.randn(1,num_head,sqlen,head_dim,requires_grad=True, device="cuda", dtype=input_type)
    dy = 0.1*torch.randn_like(Q, dtype=torch.float32, device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    def fwd():
        if use_triton:
            return FlashAttentionTriton.apply(Q,K,V,True)
        else:
            return scaled_dot_product_attention(Q,K,V,is_causal=True)
    def flash_forward_backward():
        # print("q.shape",q.shape)
        o = fwd()
        loss = o.sum()
        loss.backward()
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=500, warmup=200)
    elif mode == "backward":
        y = fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(fn=lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[Q,K,V], rep=10000, warmup=1000)
    elif mode == "forward_backward":
        ms, min_ms, max_ms = triton.testing.do_bench(flash_forward_backward, quantiles=quantiles,
                                                     grad_to_none=[Q,K,V], rep=10000, warmup=1000)
    func_str = "triton" if use_triton else "torch"
    print(f"{mode} sqlen: {sqlen} head_dim: {head_dim} {func_str} mid: {ms} ms max: {max_ms} min: {min_ms}")
    stdout.flush()

def warmup_nvtx(sqlen, head_dim, num_head):
    torch.set_float32_matmul_precision('high')
    input_type = torch.float16
    Q  = torch.randn(1,num_head, sqlen,head_dim,requires_grad=True, device="cuda", dtype=input_type)
    K  = torch.randn(1,num_head, sqlen,head_dim,requires_grad=True, device="cuda", dtype=input_type)
    V  = torch.randn(1,num_head, sqlen,head_dim,requires_grad=True, device="cuda", dtype=input_type)
    dy = 0.1*torch.randn_like(Q, dtype=torch.float32)
    attention_mask = None
    def fwd():
        # return FlashAttentionTriton.apply(Q,K,V,False)
        return _attention.apply(Q,K,V,False,1/math.sqrt(head_dim))
        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        #     print("here")
        # return scaled_dot_product_attention(Q,K,V)
    y = fwd()
    for i in range(5):
        y.backward(dy, retain_graph=True)
        torch.cuda.synchronize()
        Q.grad.zero_()
        K.grad.zero_()
        V.grad.zero_()
    torch.cuda.synchronize()
    # torch.cuda.memory._record_memory_history(max_entries=1000000)
    with nvtx.range("backward"):
        for i in range(5):
            y.backward(dy, retain_graph=True)
            torch.cuda.synchronize()
            Q.grad.zero_()
            K.grad.zero_()
            V.grad.zero_()
    # torch.cuda.memory._dump_snapshot(f"memory_snapshot.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)
    print(f"sqlen: {sqlen}, head_dim: {head_dim}")
    # print(flash_fwd_kernel.best_config)
    # print(flash_bwd_kernel_fast_dKV.best_config)
    # print(flash_bwd_kernel_fast_dQ.best_config)

def benchmark_flash_att():
    torch.set_float32_matmul_precision('high')
    sqlens = [16384]
    head_dims = [64]
    num_head = 16
    # sqlens = [65536]
    # head_dims = [128]
    modes = ["forward","backward"]
    modes = ["forward_backward"]
    for sqlen in sqlens:
        for head_dim in head_dims:
            warmup_nvtx(sqlen, head_dim, num_head)
            # for mode in modes:
            #     triton_bench(False, sqlen, head_dim, num_head, mode)
            #     triton_bench(True, sqlen, head_dim, num_head, mode)

if __name__ == "__main__":
    # test_triton()
    benchmark_flash_att()
    # best_config = flash_fwd_kernel.best_config
    # print(best_config)