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
        batch,n_head, sqlen, d = Q.shape
        T_q = sqlen // B_q
        T_k = K.shape[-2] // B_k
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((batch, n_head, sqlen), device=Q.device) ## (batch, sqlen)
        for i in range(T_q):
            Q_i = Q[:,:,i*B_q:(i+1)*B_q,:]
            O_ij = torch.zeros_like(Q_i, device=Q.device)
            l_ij = torch.zeros((batch,n_head, B_q, 1), device=Q.device)
            m_ij = torch.full((batch,n_head, B_q, 1), -torch.inf, device=Q.device)    ## negetive infinity of shape b_q
            new_m_ij = torch.empty_like(m_ij, device=Q.device)
            for j in range(T_k):
                K_j = K[:,:,j*B_k:(j+1)*B_k,:]
                V_j = V[:,:,j*B_k:(j+1)*B_k,:]
                S_ij = einsum(Q_i, K_j, "... B_q d, ... B_k d -> ... B_q B_k")/math.sqrt(d)
                if is_causal:
                    offset_q = torch.arange(B_q)[:,None]
                    offset_k = torch.arange(B_k)[None,:]
                    mask = (offset_q + i*B_q) < (offset_k + j*B_k)
                    mask = mask.to(S_ij.device)
                    S_ij = S_ij + mask * (-1e6)
                # print("S_i:", S_ij)
                new_m_ij = torch.max(m_ij, reduce(S_ij, "... B_q B_k -> ... B_q 1", "max"))
                P_ij = torch.exp(S_ij-new_m_ij)
                tmp = torch.exp(m_ij-new_m_ij)
                l_ij = tmp * l_ij + reduce(P_ij, "... B_q B_k -> ... B_q 1", "sum")
                O_ij = tmp * O_ij + einsum(P_ij, V_j.float(), "... B_q B_k, ... B_k d -> ... B_q d")
                m_ij = new_m_ij
            O[:,:,i*B_q:(i+1)*B_q,:] = O_ij / l_ij
            L[:,:,i*B_q:(i+1)*B_q] = (m_ij + torch.log(l_ij)).squeeze(-1)
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
        S = einsum(Q,K,"... N_q d, ... N_k d -> ... N_q N_k")*1/math.sqrt(d)
        # print("Q:", Q)
        # print("K:", K)
        
        if is_causal:
            mask = torch.arange(S.shape[-2])[None,:] > torch.arange(S.shape[-1])[:,None]
            mask = mask.to(S.device)
            mask = torch.where(mask, -1e6, 0)
            S += mask
        # print("S:", S)
        P = torch.exp(S-L.unsqueeze(-1))
        # print("L:", L)
        grad_V = einsum(P, grad_out,"... N_q N_k, ... N_q d -> ... N_k d")
        grad_P = einsum(grad_out, V.float(), "... N_q d, ... N_k d -> ... N_q N_k")
        
        D = reduce(O*grad_out, "... N_q d -> ... N_q 1", "sum")
        # print("D:", D)
        grad_S = P * (grad_P - D)/math.sqrt(d)
        # print("grad_S:", grad_S)
        grad_Q = einsum(grad_S,K.float(),"... N_q N_k, ... N_k d -> ... N_q d")
        # print("grad_Q:", grad_Q)
        grad_K = einsum(grad_S,Q.float(),"... N_q N_k, ... N_q d -> ... N_k d")
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

@triton.jit
def flash_fwd_kernel_inner(
    Q, K_ptr, V_ptr,
    O_i, m_i, l_i,
    query_tile_index,
    num_batch, num_head, offset_kv, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    STAGE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    K_desc = tl.make_block_ptr(
        K_ptr + offset_kv,
        shape=(num_batch * num_head * N_KEYS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_desc = tl.make_block_ptr(
        V_ptr + offset_kv,
        shape=(num_batch * num_head * N_KEYS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    if STAGE == 1:  # off-band
        lo, hi = 0, query_tile_index
    elif STAGE == 2:  # on-band
        lo, hi = query_tile_index, (query_tile_index + 1)
    else:  # non-causal
        lo, hi = 0, N_KEYS // K_TILE_SIZE
    new_m_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    # tl.device_print("lo", lo)
    # tl.device_print("hi", hi)
    for i in range(lo,hi):
        K_i = tl.load(K_desc, boundary_check=(0,1), padding_option="zero")
        V_i = tl.load(V_desc, boundary_check=(0,1), padding_option="zero")
        # tl.device_print("Q", Q)
        S_i = tl.dot(Q, tl.trans(K_i)) * scale
        # tl.device_print("S_i", S_i)
        if STAGE == 2:
            mask = tl.arange(0, Q_TILE_SIZE)[:,None] + query_tile_index * Q_TILE_SIZE < tl.arange(0, K_TILE_SIZE)[None,:] + i * K_TILE_SIZE
            S_i = tl.where(mask, -float("inf"), S_i)
        new_m_i = tl.maximum(m_i, tl.max(S_i, axis=1, keep_dims=True))
        P_i = tl.exp2(S_i-new_m_i)
        tmp = tl.exp2(m_i-new_m_i)
        l_i = tmp * l_i + tl.sum(P_i, axis=1, keep_dims=True)
        # tl.static_print("P_i", P_i)
        # tl.static_print("V_i", V_i)
        O_i = tmp * O_i + tl.dot(P_i.to(INPUT_DTYPE), V_i)
        m_i = new_m_i
        K_desc = K_desc.advance((K_TILE_SIZE, 0))
        V_desc = V_desc.advance((K_TILE_SIZE, 0))
    # tl.device_print("O_i", O_i)
    # tl.device_print("l_i", l_i)
    # tl.device_print("m_i", m_i)
    return O_i, l_i, m_i

@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8, num_stages=2),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8, num_stages=4),
        # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8), ### TODO: support Q_TILE_SIZE != K_TILE_SIZE
        # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 32}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 256, 'K_TILE_SIZE': 256}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=16),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=16),
    ],
    key=['D', 'N_QUERIES', 'N_KEYS'],   # 🔑 关键参数
)
@triton.heuristics(
    {
        'Q_TILE_SIZE': lambda args: min(args['Q_TILE_SIZE'], args['N_QUERIES']),
        'K_TILE_SIZE': lambda args: min(args['K_TILE_SIZE'], args['N_KEYS']),
    }
)
### grid should be (T_q,, batch_size), use autotune
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    num_batch, num_head, 
    N_QUERIES, N_KEYS,
    scale, o_dim,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    offset_o = batch_index * N_QUERIES * D + query_tile_index * Q_TILE_SIZE * D
    offset_l = batch_index * N_QUERIES + query_tile_index * Q_TILE_SIZE
    offset_kv = batch_index * N_KEYS * D
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_desc = tl.make_block_ptr(
        Q_ptr + offset_o,
        shape=(o_dim, D), # the shape is wrong here
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_desc = tl.make_block_ptr(
        O_ptr + offset_o,
        shape=(o_dim, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_desc = tl.make_block_ptr(
        L_ptr + offset_l,
        shape=(o_dim,),
        strides=(1,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    # tl.static_print("Q_TILE_SIZE", Q_TILE_SIZE)
    Q = tl.load(Q_desc, boundary_check=(0,1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE, 1), -float("inf"), dtype=tl.float32)
    if is_causal:
        O_i, l_i, m_i = flash_fwd_kernel_inner(
            Q, K_ptr, V_ptr, O_i, m_i, l_i, query_tile_index, num_batch, num_head, offset_kv, N_KEYS, scale, D, Q_TILE_SIZE, K_TILE_SIZE, 1, INPUT_DTYPE
        )
        O_i, l_i, m_i = flash_fwd_kernel_inner(
            Q, K_ptr, V_ptr, O_i, m_i, l_i, query_tile_index, num_batch, num_head, offset_kv + query_tile_index * K_TILE_SIZE * D, N_KEYS, scale, D, Q_TILE_SIZE, K_TILE_SIZE, 2, INPUT_DTYPE
        )
    else:
        O_i, l_i, m_i = flash_fwd_kernel_inner(
            Q, K_ptr, V_ptr, O_i, m_i, l_i, query_tile_index, num_batch, num_head, offset_kv, N_KEYS, scale, D, Q_TILE_SIZE, K_TILE_SIZE, 3, INPUT_DTYPE
        )
    # tl.device_print("out O_i", O_i)
    # tl.device_print("out l_i", l_i)
    # tl.device_print("out m_i", m_i)
    O_i = O_i / l_i
    tl.store(O_desc, O_i.to(INPUT_DTYPE))
    tl.store(L_desc, (m_i + tl.log2(l_i)).reshape(Q_TILE_SIZE))

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
            mask = tl.where(mask, 0, -float("inf"))
            S_i = S_i + mask
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
    Q_ptr, K, V,
    L_ptr, D_ptr,
    dO_ptr, dK, dV,
    offset_batch, offset_l,
    lo: tl.constexpr, hi: tl.constexpr, 
    MASK: tl.constexpr,
    N_CTX: tl.constexpr,
    scale: tl.constexpr, ## 1/(sqrt(d) * ln2)
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    kv_tile_index = tl.program_id(0)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + offset_batch,
        shape=(N_CTX, d),
        strides=(d, 1),
        offsets=(lo*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + offset_l,
        shape=(N_CTX,),
        strides=(1,),
        offsets=(lo*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + offset_batch,
        shape=(N_CTX, d),
        strides=(d, 1),
        offsets=(lo*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1,0)
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + offset_l,
        shape=(N_CTX, ),
        strides=(1, ),
        offsets=(lo*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )
    for i in range(lo, hi):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        # tl.device_print("D_i", D_i)
        # tl.device_print("Q_i", Q_i)
        # tl.device_print("K", K)
        S_i = tl.dot(Q_i, tl.trans(K), out_dtype=tl.float32) * scale
        if MASK:
            mask = tl.arange(0, Q_TILE_SIZE)[:,None] + i * Q_TILE_SIZE < tl.arange(0, K_TILE_SIZE)[None,:] + kv_tile_index * K_TILE_SIZE
            S_i = tl.where(mask, -float("inf"), S_i)
            # tl.device_print("i:", i)
        # tl.device_print("S_i", S_i)
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        L_i = tl.expand_dims(L_i,1)
        # tl.device_print("L_i",L_i)
        # tl.device_print("S_i-L_i",S_i-L_i)
        P_i = tl.exp2(S_i-L_i)
        # tl.device_print("P_i",P_i)
        # tl.device_print("dO_i",dO_i)
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        dV += tl.dot(tl.trans(P_i), dO_i, out_dtype=tl.float32)
        # tl.device_print("tmp",tmp)
        # tl.device_print("dV",dV)
        dP_i = tl.dot(dO_i.to(INPUT_DTYPE), tl.trans(V), out_dtype=tl.float32)
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        dS_i = P_i * (dP_i - tl.expand_dims(D_i,1))
        # tl.device_print("dS_i",dS_i)
        dK += tl.dot(tl.trans(dS_i).to(INPUT_DTYPE),Q_i, out_dtype=tl.float32)
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, ))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
    return dK, dV


@triton.jit
def flash_bwd_kernel_fast_dQ(
    Q, K_ptr, V_ptr,
    L, D,
    dO,dQ,
    offset_batch,
    lo, hi,
    N_CTX: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    MASK: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    tile_index = tl.program_id(0)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + offset_batch,
        shape=(N_CTX, d),
        strides=(d, 1),
        offsets=(lo*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + offset_batch,
        shape=(N_CTX, d),
        strides=(d, 1),
        offsets=(lo*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    for i in range(lo, hi):
        K_i = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_i = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        S_i = tl.dot(Q, tl.trans(K_i)) * scale
        # tl.device_print("S_i",S_i)
        if MASK:
            mask = tl.arange(0, Q_TILE_SIZE)[:,None] + tile_index * Q_TILE_SIZE < tl.arange(0, K_TILE_SIZE)[None,:] + i * K_TILE_SIZE
            S_i = tl.where(mask, -float("inf"), S_i)
        # tl.device_print("L_i",L_i)
        # tl.device_print("S_i-L_i",S_i-L_i)
        P_i = tl.exp2(S_i-L)
        # tl.device_print("P_i",P_i)
        # tl.device_print("dO_i",dO_i)
        # tl.device_print("dV",dV)
        dP_i = tl.dot(dO, tl.trans(V_i))
        dS_i = P_i * (dP_i - D)
        dQ += tl.dot(dS_i.to(INPUT_DTYPE), K_i)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    return dQ


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
    key=['d', 'N_CTX'],   # 🔑 关键参数
)

@triton.heuristics(
    {
        'Q_TILE_SIZE': lambda args: min(args['Q_TILE_SIZE'], args['N_CTX']),
        'K_TILE_SIZE': lambda args: min(args['K_TILE_SIZE'], args['N_CTX']),
    }
)
@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    N_CTX: tl.constexpr, ### only support N_Q=N_KV
    scale: tl.constexpr, ## 1/(sqrt(d) * ln2)
    scale2: tl.constexpr, ## 1/(sqrt(d))
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    offset_batch = batch_index * N_CTX * d
    offset_l = batch_index * N_CTX
    K_desc = tl.make_block_ptr(
        K_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_desc = tl.make_block_ptr(
        V_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    dK_desc = tl.make_block_ptr(
        dK_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    dV_desc = tl.make_block_ptr(
        dV_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*K_TILE_SIZE,0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    # tl.device_print("HERE")
    K = tl.load(K_desc, boundary_check=(0,1), padding_option="zero")
    V = tl.load(V_desc, boundary_check=(0,1), padding_option="zero")
    dK = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    if is_causal:
        dK,dV = flash_bwd_kernel_fast_dKV(
            Q_ptr, K, V,
            L_ptr, D_ptr,
            dO_ptr, dK, dV,
            offset_batch, offset_l,
            tile_index+1, tl.cdiv(N_CTX, Q_TILE_SIZE),
            0,
            N_CTX,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            INPUT_DTYPE,
        )
        # tl.device_print("dV1", dV)
        dK,dV = flash_bwd_kernel_fast_dKV(
            Q_ptr, K, V,
            L_ptr, D_ptr,
            dO_ptr, dK, dV,
            offset_batch, offset_l,
            tile_index, tile_index+1,
            1,
            N_CTX,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            INPUT_DTYPE,
        )
        # tl.device_print("dV2", dV)
    else:
        dK,dV = flash_bwd_kernel_fast_dKV(
            Q_ptr, K, V,
            L_ptr, D_ptr,
            dO_ptr, dK, dV,
            offset_batch, offset_l,
            0, tl.cdiv(N_CTX, Q_TILE_SIZE),
            0,
            N_CTX,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            INPUT_DTYPE,
        )
    tl.store(dK_desc, dK * scale2)
    tl.store(dV_desc, dV)
    Q_desc = tl.make_block_ptr(
        Q_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    dQ_desc = tl.make_block_ptr(
        dQ_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    D_desc = tl.make_block_ptr(
        D_ptr + offset_l,
        shape=(N_CTX,),
        strides=(1,),
        offsets=(tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0, ),
    )
    L_desc = tl.make_block_ptr(
        L_ptr + offset_l,
        shape=(N_CTX,),
        strides=(1,),
        offsets=(tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_desc = tl.make_block_ptr(
        dO_ptr + offset_batch,
        shape=(N_CTX,d),
        strides=(d,1),
        offsets=(tile_index*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    Q = tl.load(Q_desc, boundary_check=(0,1), padding_option="zero")
    D = tl.load(D_desc, boundary_check=(0,), padding_option="zero")
    D = tl.expand_dims(D,1)
    dQ = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)
    L = tl.load(L_desc, boundary_check=(0,), padding_option="zero")
    L = tl.expand_dims(L,1)
    dO = tl.load(dO_desc, boundary_check=(0,1), padding_option="zero").to(INPUT_DTYPE)
    if is_causal:
        dQ = flash_bwd_kernel_fast_dQ(
            Q, K_ptr, V_ptr,
            L, D, 
            dO, dQ,
            offset_batch,
            0,tile_index,
            N_CTX,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            0,
            INPUT_DTYPE,
        )
        dQ = flash_bwd_kernel_fast_dQ(
            Q, K_ptr, V_ptr,
            L, D, 
            dO, dQ,
            offset_batch,
            tile_index,tile_index+1,
            N_CTX,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            1,
            INPUT_DTYPE,
        )
    else:
        dQ = flash_bwd_kernel_fast_dQ(
            Q, K_ptr, V_ptr,
            L, D, 
            dO, dQ,
            offset_batch,
            0,tl.cdiv(N_CTX, Q_TILE_SIZE),
            N_CTX,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            0,
            INPUT_DTYPE,
        )
    tl.store(dQ_desc, dQ * scale2)
    

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # print(Q.shape)
        batch, num_head, num_q, d = Q.shape
        num_k = K.shape[2]
        dtype_map = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }
        grid = lambda meta: (
            triton.cdiv(num_q, meta['Q_TILE_SIZE']),
            batch * num_head,
        )
        input_dtype_triton = dtype_map.get(Q.dtype, tl.float32)
        O = torch.empty_like(Q)
        L = torch.empty((batch, num_head, num_q), device=Q.device, dtype=torch.float32)
        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            batch, num_head,
            N_QUERIES=num_q, N_KEYS=num_k,
            scale=1.44269504/math.sqrt(d),
            o_dim=batch * num_head * num_q,
            D=d,
            is_causal=is_causal,
            INPUT_DTYPE=input_dtype_triton,
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
        batch, n_head, n_ctx, d = Q.shape
        n_keys = K.shape[-2]
        assert n_ctx == n_keys, "n_queries must be equal to n_keys"
        dtype_map = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }
        grid = lambda meta: (
            triton.cdiv(n_ctx, meta['Q_TILE_SIZE']),
            batch * n_head,
        )
        input_dtype_triton = dtype_map.get(Q.dtype, tl.float32)
        D = reduce(O.float() * grad_out, "... N_ctx d -> ... N_ctx", "sum")
        dQ = torch.empty_like(Q, device=Q.device, dtype=torch.float32)
        dK = torch.empty_like(K, device=K.device, dtype=torch.float32)
        dV = torch.empty_like(V, device=V.device, dtype=torch.float32)
        flash_bwd_kernel[grid](
            Q,K,V,
            L,D,
            grad_out,dQ,dK,dV,
            n_ctx,
            scale=1.44269504/math.sqrt(d),
            scale2 = 1/math.sqrt(d),
            d=d,
            is_causal=is_causal,
            INPUT_DTYPE=input_dtype_triton,
            ### autotune, Q_TILE_SIZE and K_TILE_SIZE will be replaced by the best config
        )
        return dQ,dK,dV,None

def test_triton():
    ground_truth = scaled_dot_product_attention
    pytorch_func = FlashAttentionTorch.apply
    triton_func = FlashAttentionTriton.apply
    input_type = torch.bfloat16
    Q = torch.randn(1,1,16384,16, requires_grad=True,device="cuda", dtype=input_type)
    K = torch.randn(1,1,16384,16, requires_grad=True,device="cuda", dtype=input_type)
    V = torch.randn(1,1,16384,16, requires_grad=True,device="cuda", dtype=input_type)
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        # O_pytorch = pytorch_func(Q, K, V, True)
        O_pytorch = ground_truth(Q, K, V, None, 0, True)
    dO = torch.randn_like(O_pytorch, dtype=torch.float32)
    print("O_pytorch.dtype:", O_pytorch.dtype)
    print("dO.dtype:", dO.dtype)
    O_pytorch.backward(dO)
    Q_grad_pytorch = Q.grad.clone()
    K_grad_pytorch = K.grad.clone()
    V_grad_pytorch = V.grad.clone()
    # print("K_grad_pytorch:",K_grad_pytorch.flatten()[:10])
    # print("V_grad_pytorch:",V_grad_pytorch.flatten()[:10])
    # print("Q_grad_pytorch:",Q_grad_pytorch)
    input_type = torch.bfloat16
    Q.grad.zero_()
    K.grad.zero_()
    V.grad.zero_()
    Q = Q.detach().to(input_type).requires_grad_(True)
    K = K.detach().to(input_type).requires_grad_(True)
    V = V.detach().to(input_type).requires_grad_(True)
    O_triton = triton_func(Q, K, V, True)
    O_triton.backward(dO)
    Q_grad_triton = Q.grad.clone()
    K_grad_triton = K.grad.clone()
    V_grad_triton = V.grad.clone()
    # print("K_grad_triton:",K_grad_triton.flatten()[:10])
    # print("V_grad_triton:",V_grad_triton.flatten()[:10])
    # print("Q_grad_triton:",Q_grad_triton)
    # print("O_pytorch:", O_pytorch)
    # print("O_triton:", O_triton)
    print("output max error:", torch.max(torch.abs(O_pytorch - O_triton)))
    print("Q grad max error:", torch.max(torch.abs(Q_grad_pytorch - Q_grad_triton)))
    print("K grad max error:", torch.max(torch.abs(K_grad_pytorch - K_grad_triton)))
    print("V grad max error:", torch.max(torch.abs(V_grad_pytorch - V_grad_triton)))
    print(flash_fwd_kernel.best_config)
    print(flash_bwd_kernel.best_config)
    # print("output max error:", torch.allclose(O_pytorch, O_triton, atol=1e-2, rtol=1e-3))
    # print("Q grad max error:", torch.allclose(Q_grad_pytorch, Q_grad_triton, atol=1e-2, rtol=1e-3))
    # print("K grad max error:", torch.allclose(K_grad_pytorch, K_grad_triton, atol=1e-2, rtol=1e-3))
    # print("V grad max error:", torch.allclose(V_grad_pytorch, V_grad_triton, atol=1e-2, rtol=1e-3))

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
        return FlashAttentionTriton.apply(Q,K,V,True)
        # return _attention.apply(Q,K,V,False,1/math.sqrt(head_dim))
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
    test_triton()
    # benchmark_flash_att()
    # best_config = flash_fwd_kernel.best_config
    # print(best_config)