import torch 
import torch.nn as nn 
from typing import Tuple 

def compute_freq(dim:int, seq_len:int, theta_0:float):
    freqs = 1 / (theta_0**((torch.arange(0, dim, 2)).float()/dim))
    m = torch.arange(0, seq_len, device=freqs.device)
    #m*theta for each mth pos fine the freq nxm so outer product 
    mtheta = torch.outer(m, freqs).float() #(seq_len, dim//2)
    return torch.polar(torch.ones_like(mtheta), mtheta)#complex polar cooordinate e^mtheta 

def apply_rope(
        xq:torch.Tensor,
        xk:torch.Tensor,
        mtheta_complex:torch.Tensor
):
    #RoPE applies the same positional rotation along the sequence axis, not across the batch.
    batch, seq_len, dim = xq.shape 
    xq = xq.permute(1, 0, 2)
    seq_len, batch, dim = xq.shape 
    xk = xk.permute(1, 0, 2) #seq_len, batch, dim 
    seq_len, dim_2 = mtheta_complex.shape 

    mtheta_complex = mtheta_complex.to(xq.device)
    #we are applying rope in the attention stage so it invloves in RoPE(Q, K)
    # Convert xq to complex by reshaping last dim into pairs (real, imag): shape -> (seq_len, batch, dim // 2)

    #[1, 0, 0, 1] -> [[1,0], [0, 1]] =  [[x1, x2], [x3, x4]]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    mtheta_complex = mtheta_complex.unsqueeze(dim=1) # seq_len, 1, dim//2 

    #Rope(x1, x2)
    # Apply rotation (complex multiplication) for queries: shape -> (seq_len, batch, dim // 2, 2) → flatten → (seq_len, batch, dim)
    xq_out = torch.view_as_real(xq_ * mtheta_complex).flatten(-2)
    xk_out = torch.view_as_real(xk_ * mtheta_complex).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk) 

