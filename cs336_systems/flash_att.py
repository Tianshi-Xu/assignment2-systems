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
                O_ij = tmp * O_ij + einsum(P_ij, V_j, "batch B_q B_k, batch B_k d -> batch B_q d")
                m_ij = new_m_ij
            O[:,i*B_q:(i+1)*B_q,:] = O_ij / l_ij
            L[:,i*B_q:(i+1)*B_q] = (m_ij + torch.log(l_ij)).squeeze(-1)
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        # print("O.shape", O.shape)
        # print("L.shape", L.shape)
        return O

    @staticmethod
    def backward(ctx, grad_out):
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
        grad_P = einsum(grad_out, V, "batch N_q d, batch N_k d -> batch N_q N_k")
        # print("grad_V:", grad_V)
        D = reduce(O*grad_out, "batch N_q d -> batch N_q 1", "sum")
        grad_S = P * (grad_P - D)
        grad_Q = einsum(grad_S,K,"batch N_q N_k, batch N_k d -> batch N_q d")/math.sqrt(d)
        # print("grad_Q:", grad_Q)
        grad_K = einsum(grad_S,Q,"batch N_q N_k, batch N_q d -> batch N_k d")/math.sqrt(d)
        return grad_Q, grad_K, grad_V, None

def scaled_dot_product_attention(
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

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=2),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=4, num_stages=3),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=8),
    ],
    key=['D', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
)
@triton.heuristics(
    {
        'K_TILE_SIZE': lambda k: k['Q_TILE_SIZE'],
    }
)
### grid should be (T_q,, batch_size), use autotune
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
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
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE, 1), -float("inf"), dtype=tl.float32)
    new_m_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_i = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_i = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        S_i = tl.dot(Q, tl.trans(K_i), out_dtype=tl.float32)
        S_i = S_i * scale
        if is_causal:
            offset_q = tl.arange(0,Q_TILE_SIZE)
            offset_q = tl.expand_dims(offset_q,1)
            offset_k = tl.arange(0,K_TILE_SIZE)
            offset_k = tl.expand_dims(offset_k,0)
            mask = (offset_q + query_tile_index * Q_TILE_SIZE) >= (offset_k + i * K_TILE_SIZE)
            mask = tl.where(mask, 0, -1e6)
            S_i = S_i + mask 
        new_m_i = tl.maximum(m_i, tl.max(S_i, axis=1, keep_dims=True))
        P_i = tl.exp(S_i-new_m_i)
        tmp = tl.exp(m_i-new_m_i)
        l_i = tmp * l_i + tl.sum(P_i, axis=1, keep_dims=True)
        O_i = tmp * O_i + tl.dot(P_i, V_i)
        m_i = new_m_i
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O_i = O_i / l_i
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, (m_i + tl.log(l_i)).reshape(Q_TILE_SIZE))

@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 256}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 128}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 256}, num_warps=16),
    ],
    key=['d', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
)

@triton.heuristics(
    {
        'K_TILE_SIZE': lambda k: k['Q_TILE_SIZE'],
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
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
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
    dK = tl.load(dK_block_ptr, boundary_check=(0,1), padding_option="zero")
    dV = tl.load(dV_block_ptr, boundary_check=(0,1), padding_option="zero")
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
            mask = tl.where(mask, 0, -1e6)
            S_i = S_i + mask
            # tl.device_print("mask", mask)
        L_i = tl.expand_dims(L_i,1)
        # tl.device_print("L_i",L_i)
        # tl.device_print("S_i-L_i",S_i-L_i)
        P_i = tl.exp(S_i-L_i)
        # tl.device_print("P_i",P_i)
        dV += tl.dot(tl.trans(P_i), dO_i, out_dtype=tl.float32)
        # tl.device_print("dV",dV)
        dP_i = tl.dot(dO_i, tl.trans(V))
        dS_i = P_i * (dP_i - tl.expand_dims(D_i,1)) * scale
        tmp = tl.dot(dS_i, K, out_dtype=tl.float32)
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
        dK += tl.dot(tl.trans(dS_i),Q_i)
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, ))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
    tl.store(dK_block_ptr, dK)
    tl.store(dV_block_ptr, dV)

@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64}, num_warps=8),
    ],
    key=['d', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
)

@triton.heuristics(
    {
        'K_TILE_SIZE': lambda k: k['Q_TILE_SIZE'],
    }
)
@triton.jit
def flash_bwd_kernel_fast_fake(
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
    dK = tl.load(dK_block_ptr, boundary_check=(0,1), padding_option="zero")
    dV = tl.load(dV_block_ptr, boundary_check=(0,1), padding_option="zero")
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        dQ_i = tl.load(dQ_block_ptr, boundary_check=(0,1), padding_option="zero")
        S_i = tl.dot(Q_i, tl.trans(K), out_dtype=tl.float32) * scale
        # tl.device_print("S_i",S_i)
        if is_causal:
            offset_q = tl.arange(0,Q_TILE_SIZE)
            offset_q = tl.expand_dims(offset_q,1)
            offset_k = tl.arange(0,K_TILE_SIZE)
            offset_k = tl.expand_dims(offset_k,0)
            mask = (offset_q + i * Q_TILE_SIZE) >= (offset_k + kv_tile_index * K_TILE_SIZE)
            mask = tl.where(mask, 0, -1e6)
            S_i = S_i + mask
        L_i = tl.expand_dims(L_i,1)
        P_i = tl.exp(S_i-L_i)
        dV += tl.dot(tl.trans(P_i), dO_i, out_dtype=tl.float32)
        dP_i = tl.dot(dO_i, tl.trans(V))
        dS_i = P_i * (dP_i - tl.expand_dims(D_i,1)) * scale
        tmp = tl.dot(dS_i, K, out_dtype=tl.float32)
        dQ_i += tmp
        dK += tl.dot(tl.trans(dS_i),Q_i)
        # tl.store(dQ_block_ptr, dQ_i)
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
    L_ptr, dO_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_do, stride_dod,
    stride_lb, stride_lq,
    stride_D,
    N_QUERIES, N_KEYS,
    scale,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pass

@triton.jit
def flash_bwd_kernel_fast_dQ(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, dO_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,  ### stride for batch, sqlen, d
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_do, stride_dod,
    stride_lb, stride_lq,
    stride_D,
    N_QUERIES, N_KEYS,
    scale,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pass

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        grid = lambda meta: (
            triton.cdiv(sqlen, meta['Q_TILE_SIZE']),
            batch,
        )
        batch, sqlen, d = Q.shape
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((batch, sqlen), device=Q.device)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=sqlen, N_KEYS=K.shape[1],
            scale=1/math.sqrt(d),
            D=d,
            is_causal=is_causal,
            ### autotune, Q_TILE_SIZE and K_TILE_SIZE will be replaced by the best config
        )
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        return O
    
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
    def backward(ctx: Any, grad_out: Any) -> Any:
        Q,K,V,L,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        grid = lambda meta: (
            triton.cdiv(n_keys, meta['K_TILE_SIZE']),
            batch,
        )
        D = reduce(O*grad_out, "batch N_q d -> batch N_q", "sum")
        dQ = torch.zeros_like(Q, device=Q.device)
        dK = torch.zeros_like(K, device=K.device)
        dV = torch.zeros_like(V, device=V.device)
        flash_bwd_kernel_fast_fake[grid](
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
            is_causal=is_causal
        )
        return dQ,dK,dV,None

def test_triton():
    ground_truth = scaled_dot_product_attention
    pytorch_func = FlashAttentionTorch.apply
    triton_func = FlashAttentionTriton.apply
    Q = torch.randn(1,16,16, requires_grad=True,device="cuda")
    K = torch.randn(1,16,16, requires_grad=True,device="cuda")
    V = torch.randn(1,16,16, requires_grad=True,device="cuda")
    O_pytorch = pytorch_func(Q, K, V, True)
    O_pytorch.sum().backward()
    Q_grad_pytorch = Q.grad.clone()
    K_grad_pytorch = K.grad.clone()
    V_grad_pytorch = V.grad.clone()
    print("V_grad_pytorch:",V_grad_pytorch)
    Q.grad.zero_()
    K.grad.zero_()
    V.grad.zero_()
    O_triton = triton_func(Q, K, V, True)
    O_triton.sum().backward()
    Q_grad_triton = Q.grad.clone()
    K_grad_triton = K.grad.clone()
    V_grad_triton = V.grad.clone()
    print("V_grad_triton:",V_grad_triton)
    print("q_grad:", torch.max(torch.abs(Q_grad_triton - Q_grad_pytorch)))
    print("k_grad:", torch.max(torch.abs(K_grad_triton - K_grad_pytorch)))
    print("v_grad:", torch.max(torch.abs(V_grad_triton - V_grad_pytorch)))
    # print("O_pytorch", O_pytorch.flatten()[0:10])
    # print("O_ground_truth", O_ground_truth.flatten()[0:10])
    # print("O_triton", O_triton.flatten()[0:10])
    # print("O_pytorch - O_ground_truth", (O_pytorch - O_ground_truth).abs().max())
    # print("O_triton - O_ground_truth", (O_triton - O_ground_truth).abs().max())

def triton_bench(use_triton, sqlen, head_dim, mode, samples=5, warmup_steps=5):
    torch.set_float32_matmul_precision('high')
    Q  = torch.randn(1,sqlen,head_dim,requires_grad=True, device="cuda")
    K  = torch.randn(1,sqlen,head_dim,requires_grad=True, device="cuda")
    V  = torch.randn(1,sqlen,head_dim,requires_grad=True, device="cuda")
    dy = 0.1*torch.randn_like(Q)
    quantiles = [0.5, 0.2, 0.8]
    def fwd():
        if use_triton:
            return FlashAttentionTriton.apply(Q,K,V,True)
        else:
            return scaled_dot_product_attention(Q,K,V,True)
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=500, warmup=200)
    elif mode == "backward":
        y = fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[Q,K,V], rep=2000, warmup=1000)
    func_str = "triton" if use_triton else "torch"
    print(f"{mode} sqlen: {sqlen} head_dim: {head_dim} {func_str} mid: {ms} ms max: {max_ms} min: {min_ms}")
    stdout.flush()

def warmup(sqlen, head_dim):
    torch.set_float32_matmul_precision('high')
    Q  = torch.randn(1,sqlen,head_dim,requires_grad=True, device="cuda")
    K  = torch.randn(1,sqlen,head_dim,requires_grad=True, device="cuda")
    V  = torch.randn(1,sqlen,head_dim,requires_grad=True, device="cuda")
    dy = 0.1*torch.randn_like(Q)
    def fwd():
        return FlashAttentionTriton.apply(Q,K,V,True)
        # return scaled_dot_product_attention(Q,K,V,True)
    y = fwd()
    torch.cuda.synchronize()
    with nvtx.range("backward"):
        y.backward(dy, retain_graph=True)
        torch.cuda.synchronize()
    
    # print(flash_fwd_kernel.best_config)
    print(f"sqlen: {sqlen}, head_dim: {head_dim}")
    # print(flash_bwd_kernel_fast_fake.best_config)

def benchmark_flash_att():
    torch.set_float32_matmul_precision('high')
    sqlens = [32768]
    head_dims = [16,32,64,128]
    # sqlens = [65536]
    # head_dims = [128]
    modes = ["backward"]
    for sqlen in sqlens:
        for head_dim in head_dims:
            warmup(sqlen, head_dim)
            # for mode in modes:
            #     triton_bench(True, sqlen, head_dim, mode)
            #     triton_bench(False, sqlen, head_dim, mode)

if __name__ == "__main__":
    # test_triton()
    benchmark_flash_att()
    # best_config = flash_fwd_kernel.best_config
    # print(best_config)