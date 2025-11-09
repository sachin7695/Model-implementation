import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
import sys, os
from windowed_attention import LocalWindowAttention
from rope import compute_freq, apply_rope
import tiktoken 
import time
from torch.cuda.amp import autocast, GradScaler

# hyperparameters
batch_size = 4 # independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 2500
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 150
n_embd = 768
n_head = 12
n_layer = 8
dropout = 0.2
vocab_size_new = 50257
sliding_window_size = 64
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('/home/cmi_10101/Documents/coding/pytorch/architecture-implementation/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# #tokenization on the basis of character


# ===== Tokenizer =====
# Tiktoken tokenizer
use_tiktoken = False  # toggle this

if use_tiktoken:
    tokenizer = tiktoken.get_encoding("gpt2") 
    vocab_size = 50257

    def encode(s):
        return tokenizer.encode(s)
    def decode(l):
        return tokenizer.decode(l)

else:
    # Fallback: char-level (Karapathy style)
    # tokenization on the basis of character
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text),dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

train_loader = DataLoader(CharDataset(train_data, block_size), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(CharDataset(val_data, block_size), batch_size=batch_size, drop_last=True)

def get_batch_from_loader(loader_iter, loader):
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    return x.to(device), y.to(device), loader_iter

# Then in training loop, instead of get_batch('train') you’ll do:

train_iter = iter(train_loader)
val_iter = iter(val_loader)

xb, yb, train_iter = get_batch_from_loader(train_iter, train_loader)


def get_lr(it, max_lr, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#data loading
def get_batch(split):
    data  = train_data if split=="train" else val_data 
    ix = torch.randint(len(data)-block_size, size=(batch_size,)) #[10, 30, 56, 78]
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y 


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size  # n_embed // num_heads
        self.context_length = block_size 
        self.cap_cache_to_ctx = True
        self.sliding_window_size = sliding_window_size


        self.w_key = nn.Linear(n_embd, head_size, bias = False)
        self.w_query = nn.Linear(n_embd, head_size, bias=False)
        self.w_value = nn.Linear(n_embd, head_size, bias=False)


        self.register_buffer('mask', 
                            torch.tril(torch.ones((block_size, block_size))),
                            persistent=False) # block size is context length
        self.mtheta_complex = compute_freq(dim=n_embd, seq_len=block_size, theta_0=10000)
        self.dropout = nn.Dropout(0.2)


        # for KV-cache
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.ptr_current_pos = 0

    def reset_cache(self):
        self.k_cache, self.v_cache = None, None 
        self.pt_current_pos  = 0

    def _maybe_cap_cache(self):
        """Keep only the most recent `context_length` tokens in cache (optional)."""
        if not self.cap_cache_to_ctx:
            return
        if self.k_cache is None:
            return
        cur_len = self.k_cache.size(1)
        if cur_len > self.context_length:
            keep = self.context_length
            self.k_cache = self.k_cache[:, cur_len - keep :, :]
            self.v_cache = self.v_cache[:, cur_len - keep :, :]
            # When we cap, we also need to cap the ptr so mask slicing stays valid
            self.ptr_current_pos = min(self.ptr_current_pos, self.context_length)


    def forward(self,x, use_cache=False):
        B, T, C = x.shape #batch, time-step, n_embd (channels) in inference stage for kv cache T=1

        keys_new = self.w_key(x) # (B, T, HS)
        values_new = self.w_value(x) # (B, T, HS)
        q = self.w_query(x) 


        # RoPE in attention with q and k 
        # RoPE(xq, xk) 
        
        # k = self.w_key(x)  #(B, T, head_size)
        # q = self.w_query(x) #(B, T, head_size) 
        # v = self.w_value(x)  #B, T, head_size


        # RoPE apply
        # q, keys_new = apply_rope(q, keys_new, self.mtheta_complex)
        # k, q -> T, B, C the RoPE chnaged this dimension  thats why we have permute it again to (B, T, C)
        ''' we are not applying RoPE as of now comment out below two lines to apply RoPE'''
        # keys_new = keys_new.permute(1,0, 2).contiguous() #B, T, C
        # q = q.permute(1, 0, 2).contiguous()  # B , T, C


        if use_cache and not self.training:
            ### sliding window attention ### 
            old_len = 0 if self.k_cache is None else self.k_cache.shape[1]
            if self.k_cache is None:
                self.k_cache, self.v_cache  = keys_new, values_new 
            else:
                self.k_cache = torch.cat([self.k_cache, keys_new], dim=1)
                self.v_cache = torch.cat([self.v_cache, values_new], dim=1)

            
            # Left trim to sliding window if condfigured 
            if self.sliding_window_size is not None:
                if self.k_cache.shape[1] > self.sliding_window_size :
                    self.k_cache = self.k_cache[:, -self.sliding_window_size:, :]
                    self.v_cache = self.v_cache[:, -self.sliding_window_size:, :]
            total_len = old_len + T # T = x.shape[1]
            k_len_now = self.k_cache.shape[1] 
            dropped = max(0, total_len - k_len_now) 
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
            keys, values = self.k_cache, self.v_cache 
        else:
            keys, values = keys_new, values_new # B, T, HS

        keys = keys.permute(0, 2, 1).contiguous() # B, HS, T
        attn_scores = q @ keys # B, 1, HS @ B, HS, T  -> B, 1, T


        # causal + sliding-window mask
        num_tokens_q = q.shape[1]
        num_tokens_k = keys.shape[2] 
        device = q.device 

        # Determine absolute position for q and k 
        if use_cache:
            q_start = q_start_pos_abs 
            k_start = k_start_pos_abs 
        else:
            q_start = 0
            k_start = 0 
        


        q_positions = torch.arange(q_start, q_start+num_tokens_q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start+num_tokens_k, device=device, dtype=torch.long)

        # sliding  window _width and put causal mask on that 
        win_size = num_tokens_k + 1 if self.sliding_window_size is None else (self.sliding_window_size)
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0) 
        mask_bool = (diff < 0) | (diff >= win_size)

        if use_cache and not self.training:
            ''' build causal mask static way '''
            # mask_bool = self.mask.bool()[ 
            #     self.pt_curr_pos:self.ptr_current_pos+num_tokens_q, :num_tokens_k
            # ]
            # self.ptr_current_pos += num_tokens_q 

            # past_len = self.ptr_current_pos 

            # build rectangular causal mask dynamically 

            # i = torch.arange(num_tokens_q, device=x.device).unsqueeze(1) # T, 1
            # j = torch.arange(num_tokens_k, device=x.device).unsqueeze(0) # 1, T 


            # each query i can attend to all past (past_len) + tokens up to itself (i)
            # mask_bool = (j > (past_len + i))  # True means MASK
            self.ptr_current_pos += num_tokens_q 


        else:
            # mask_bool = self.mask.bool()[:num_tokens_q, :num_tokens_k]]
            # i = torch.arange(num_tokens_q, device=x.device).unsqueeze(1)
            # j = torch.arange(num_tokens_k, device=x.device).unsqueeze(0)
            # mask_bool = (j > i)  # simple causal mask for training (no past)
            self.pt_current_pos = 0
        



        attn_scores.masked_fill(mask_bool, float("-inf"))
        scale = math.sqrt(self.head_size)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ values




        # if use_cache and not self.training:
        #     #initialize k/v cache if empty 
        #     if self.k_cache is None or self.v_cache is None:
        #         self.k_cache = torch.zeros(B, block_size, self.head_size, device=q.device) #b, T, HEAD_SIZE
        #         self.v_cache = torch.zeros(B, block_size, self.head_size, device=v.device) #B, T, HEAD_SIZE
        #         self.cache_idx = 0 

        #     #update k/v cache 
        #     if self.cache_idx + T <= block_size:
        #         self.k_cache[:, self.cache_idx:self.cache_idx+T, :] = k #passing one token at a time for inference stage 
        #         self.v_cache[:, self.cache_idx:self.cache_idx+T, :] = v 


        #     #when the cache is full it can not take the next token 
        #     # then we need to shift one position left 
        #     else:
        #         shift = self.cache_idx + T - block_size # shift = 1
        #         self.k_cache[:, :-1, :] = self.k_cache[:, 1:, :] #1 position left shift 
        #         self.v_cache[:, :-1, :] = self.v_cache[:, 1:, :] 

        #         #assign last oen to new_k and new_v
        #         self.k_cache[:, -T:, :] = k #here T =  1 one token by token 
        #         self.v_cache[:, -T:, :] = v
        #     self.cache_idx = min(self.cache_idx+T, block_size) #should not be outbound of block size or max_seq_len 


        #     #calculating affinities 
        #     #attn q*k.T/root(2)
        #     wei = q @ self.k_cache.transpose(-2, -1) * k.shape[-1]**-0.5
        #     wei = wei.masked_fill(self.tril[:T, :self.cache_idx]==0 , float("-inf"))
        #     wei = F.softmax(wei, dim=-1)
            
        #     out = wei @ self.v_cache #B, T, HS 
        # else:
        #     # ===== TRAINING MODE (or no cache requested) =====
        #     # Standard full attention computation
        #     wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        #     wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        #     wei = F.softmax(wei, dim=-1)
        #     wei = self.dropout(wei)
        #     out = wei @ v 
        


        return out

class MHA(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        #O(N**2) computation 
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))

        # O(N*W) computation 
        # self.head is already having the cumulative heads so we dont need to loop 
        # it through multiple heads
        # self.heads = nn.ModuleList(LocalWindowAttention(block_size=block_size, 
        #                                                 window_size=64, 
        #                                                 causal=True) 
        #                                                 for _ in range(num_heads)
        #                                                 )


        # self.head = LocalWindowAttention( #cumulative heads
        #     block_size=block_size, 
        #     window_size=64, 
        #     causal=True
        #     )

        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def reset_cache(self):
        """Reset KV cache in all heads"""
        # for head in self.head:
        # self.head.reset_cache() # this is for SWA 

        for head in self.heads: # in MHA as each head has reset_cache method so we looped through
            head.reset_cache()

    def forward(self, x, use_cache = False):
        out = torch.cat([h(x, use_cache = use_cache) for h in self.heads], dim=-1)
        # out = self.heads(x, use_cache=use_cache)
        out = self.dropout(self.proj(out))
        return out
    
class Expert(nn.Module):

    def __init__(self, n_embd, expansion = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion*n_embd), #(B, T, C) -> (B, T, 4*C)
            nn.ReLU(),
            nn.Linear(expansion*n_embd, n_embd) #(B, T, 4*C) -> (B, T, C) 

        )
    def forward(self, x):
        return self.net(x) 
    
#ROUTER 
class Top2gate(nn.Module):
    """Router module for selecting experts in MoE"""
    def __init__(self, n_embd, n_experts, moe_loss_coeff=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(n_embd, self.n_experts, bias=False) #logits per experts 
        self.moe_loss_coeff = moe_loss_coeff 
    
    def forward(self, x):
        #x (b, t, c)
        logits = self.gate(x) #H(X) routing matrix (b, t, n_embd) -> (b, t, n_experts)
        probs = F.softmax(logits, dim=-1) #per token distributions of routing matrix (B, T, n_experts)
        #probs[b, t, e] e=0, 1, 2 if n_experts = 3 tells how much Ei giving attention to X[:, t] token 
        top_2_vals, top_2_idx = probs.topk(k=2, dim=-1) #we chose only top 2 bets experts per token (B, T, 2)

        '''
        Discrete routing choice: We only use the best two experts per token. top2_idx[b,t,:] holds the expert IDs; 
        top2_vals holds their probabilities.
        '''
        #re-normalize the two probs to sum to 1 
        denom = top_2_vals.sum(dim=-1, keepdim=True) #(B, T, 1) 
        gate_probs_top_2 = top_2_vals / denom #(B, T, 2) normalized 

        # aux load-balancing loss (simple Switch-style)
        importance = probs.mean(dim=(0, 1)) #mean over batch and time steps (N_experts, )
        #importance[e]: On average (over batch & time), how much probability the router assigns to expert e.

        #fraction of tokens dispatched to expert
        top1_choice_expert_idx = probs.argmax(dim = -1) #(B, T) idx[1, 0, 2, 0] n_expert = 3
        one_hot = F.one_hot(top1_choice_expert_idx, num_classes=self.n_experts).float() #(B, T, n_experts)

        #fraction of tokens choosing E1 as top1 choice 
        load = one_hot.mean(dim=(0, 1)) #avergaed over batch and time 

        aux_loss = self.n_experts * torch.sum(importance * load) * self.moe_loss_coeff #sigma FiPi 
        return gate_probs_top_2, top_2_idx, aux_loss
    

class MoEFFN(nn.Module):
    def __init__(self,
                n_embd, 
                n_experts = 3, 
                expansion = 4, 
                dropout = 0.2,
                moe_loss_coeff = 0.01, #(lambda hyper parameter tuning
                ):
        super().__init__()
        
        self.n_experts = n_experts
        self.experts = nn.ModuleList([Expert(n_embd, expansion) for _ in range(self.n_experts)])
        self.router = Top2gate(n_embd, n_experts=self.n_experts, moe_loss_coeff=moe_loss_coeff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_probs_top_2, top_2_idx, aux_loss = self.router(x) # (B,T,2), (B,T,2), scalar

        #decide for each token which 2 expert to use and by how much

        all_outs = torch.stack([e(x) for e in self.experts], dim=-1)
        #e1(x) -> (B, T, C)
        #e2(x) -> (B, T, C)
        #e3(x) -> (B, T, C)
        #.
        #.
        #en(x) -> (B, T, C) 
        #stacking them all (B, T, C, E)

        # build a sparse weight mask with only top-2 per token
        B, T, _ = x.shape
        E = self.n_experts 
        mask  = torch.zeros(size=(B, T, E), device=x.device, dtype=x.dtype) #(B, T, E)
        # fill only the chosen experts
        mask.scatter_(dim=-1, index=top_2_idx, src=gate_probs_top_2)
        # combine: (B,T,C,E) * (B,T,1,E) -> (B,T,C)
        y = (all_outs * mask.unsqueeze(-2)).sum(dim=-1)
        y = self.dropout(y)

        return y, aux_loss 
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x, 3))
        ))


class Feedforward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            
            # nn.ReLU(),
            # Replace RELU with GELU
            GELU(),

            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_head, use_moe = False, n_experts  =3, moe_loss_coeff = 0.01):
        super().__init__()
        self.head_size = n_embd // n_head 
        self.sa = MHA(num_head, self.head_size) #B, T, n_embd 

        #replace layer norm by RMS norm
        # self.ln1 = nn.LayerNorm(n_embd)
        # self.ln2 = nn.LayerNorm(n_embd)

        #RMS Norm layer
        self.norm1 = RMSNorm(dim=n_embd)
        self.norm2 = RMSNorm(dim=n_embd)


        self.use_moe = use_moe 
        if use_moe:
            self.ffn = MoEFFN(
                n_embd, n_experts, moe_loss_coeff=moe_loss_coeff
            )
        else:
            self.ffn = Feedforward(n_embd)

    # def reset_cache(self):
    #     """Reset KV cache in attention"""
    #     self.sa.reset_cache()


    def forward(self, x, use_cache = False):
        x = x+self.sa(self.norm1(x), use_cache=use_cache)

        if self.use_moe:
            y, aux = self.ffn(self.norm2(x))
            x = x+y        
        else:
            # print(self.ln2(x).shape)
            # print()
            
            y = self.ffn(self.norm2(x))
            aux = torch.tensor(0.0, device=x.device)

            x = x + y 
        return x, aux




class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.tok_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, num_head=n_head,
                   use_moe=(i % 2 == 0), n_experts=3, moe_loss_coeff=0.01
                   ) for i in range(n_layer)]
        )

        # self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) 
        self.rms_norm = RMSNorm(dim=n_embd)


        self.apply(self._init_weights)
        self.current_pos = 0

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)


    def reset_cache(self):
        """Reset KV cache in all blocks - call before generation"""
        for block in self.blocks:
            block.sa.reset_cache()
        self.current_pos = 0
        
    
    def forward(self, idx, targets = None, use_cache = False):
        B, T = idx.shape  #T-> seq_len (block size)
        aux_total = torch.tensor(0.0, device=idx.device)


        tok_embd = self.tok_embedding_table(idx) #(B, T, C)
        #added RMS norm after the tokenization embedding
        tok_embd = self.rms_norm(tok_embd) #B, T, C 

        #Pass through RMS Layer as Mistral
        # tok_emb = RMSNorm(dim=tok_embd.shape[-1]) #B, T, C 
        #we will later pass through RoPE

        # pos_embd = self.pos_embedding_table(torch.arange(T, device = device)) #(B, T, C)
        # x = tok_embd + pos_embd #(B, T, C)

        # now we dont need POS_emb here as we are passing it through RoPE 
        # in the attention layer 


        ### KV Cache ##### 
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos+T, device=idx.device, dtype=torch.long)
            self.current_pos += T 
        else:
            pos_ids = torch.arange(0, T, device=idx.device, dtype=torch.long) 
        pos_embd = self.pos_embedding_table(pos_ids).unsqueeze(0) # (T, C)

        
        x = tok_embd + pos_embd # Shape [batch_size, num_tokens, emb_size]

        for block in self.blocks:
            x, aux = block(x, use_cache=use_cache)
            aux_total += aux 

        # x = self.ln_f(x) #(B, T, C)
        x = self.rms_norm(x) #B, T, C
        logits = self.lm_head(x) #(B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss_lm = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) #logits -> (N, C) - targets-> (N,) 

            loss = loss_lm + aux_total 

        return logits, loss 
    

    def generate_simple(self, model, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):



            # crop idx to the last block_size tokens
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context


            idx_cond = idx[:, -block_size:]
            # get the predictions
            with torch.no_grad():
                logits, loss = model(idx_cond) 


            # focus only on the last time step
            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :] 


            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def generate(self, model, prompt , max_new_tokens=100, top_k=50, temperature=0.8, eos_token=None, use_cache = True):
        model.eval()
        ctx_len = block_size or model.pos_embedding_table.num_embeddings

        idx = prompt.clone()

        with torch.no_grad():
            if use_cache:
                # init cache with full prompt 
                model.reset_cache() 
                logits = model(idx[:, -ctx_len:], use_cache=True)
                if isinstance(logits, tuple):
                    logits = logits[0]  # first element is usually logits
                else:
                    logits = logits

                for _ in range(max_new_tokens):
                    if logits.dim()==3:
                        logits = logits[:, -1, :] #(B, vocab_size)
                    
                    logits = logits / temperature 
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                    # top-k filtering and re-normalization
                    topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8) 
                    # multinomial sampling
                    next_token_rel = torch.multinomial(topk_probs, 1)
                    next_token = topk_idx.gather(-1, next_token_rel)
                    idx = torch.cat((idx, next_token), dim=1)
                    if eos_token is not None and next_token.item() == eos_token:
                        break
        return idx

        # for _ in range(max_new_tokens):

        #     #token by token generation as kv-cache takes one token at a time
        #     logits, _ = model(idx[:, -1:])
        #     logits = logits[:, -1, :] / temperature

        #     # numerical safety clamp
        #     logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        #     probs = F.softmax(logits, dim=-1)


        #     topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        #     # next_token = topk_idx.gather(-1, torch.multinomial(topk_probs, 1))

        #     # top-k filtering and re-normalization
        #     topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        #     # multinomial sampling
        #     next_token_rel = torch.multinomial(topk_probs, 1)
        #     next_token = topk_idx.gather(-1, next_token_rel)

        #     idx = torch.cat((idx, next_token), dim=1)
        #     if eos_token is not None and next_token.item() == eos_token:
        #         break
        # return idx



def save_checkpoint(model, optimizer, scaler, iter, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")




def load_checkpoint(filepath, model, optimizer=None, scaler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"Resumed from iteration: {checkpoint['iter']}")
    print(f"Loss: {checkpoint['loss']:.4f}")
    return checkpoint['iter'], checkpoint["loss"]

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    torch.manual_seed(1337)
    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    print(f"\nModel has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # === Resume or Start Fresh ===
    checkpoint_dir = "./checkpoints_v5"
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_path = None  # e.g., "./checkpoints_v3/model_iter_2000.pt"
    start_iter, best_val_loss = 0, float("inf")

    if resume_path and os.path.exists(resume_path):
        start_iter, _ = load_checkpoint(resume_path, model, optimizer, scaler)
        print(f"Resuming from iteration {start_iter}")
    else:
        print("Starting training from scratch...")

    # =====================================================
    #                    TRAINING LOOP
    # =====================================================
    for iter in range(start_iter, max_iters):

        # ---- Evaluate periodically ----
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            ckpt_path = os.path.join(checkpoint_dir, f"model_iter_{iter}.pt")
            save_checkpoint(model, optimizer, scaler, iter, losses["val"], ckpt_path)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_checkpoint(model, optimizer, scaler, iter, losses["val"], best_path)
                print(f"✅ New best model! val_loss={best_val_loss:.4f}")

        # ---- Training batch ----
        xb, yb = get_batch("train")

        with autocast(dtype=torch.bfloat16):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if iter % 50 == 0:
            print(f"Iter {iter} | Loss: {loss.item():.4f}")

    # =====================================================
    #                     FINAL SAVE
    # =====================================================
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    save_checkpoint(model, optimizer, scaler, max_iters, loss.item(), final_path)
    print(f"\n✅ Training Complete | Best val_loss: {best_val_loss:.4f}")

    # =====================================================
    #                   TEXT GENERATION
    # =====================================================
    print("\n" + "=" * 50)
    print("Generating sample text...")
    start_context = "hello, i am"
    encoded = encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    with torch.no_grad():
        token_ids = model.generate(model, prompt=encoded_tensor, max_new_tokens=100)

    decoded_text = decode(token_ids[0].tolist())
    print("\n" + "=" * 50)
    print(decoded_text)
    print("=" * 50)



if __name__ == "__main__":
    main()
