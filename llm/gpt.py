import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
import sys
from windowed_attention import LocalWindowAttention
from rope import compute_freq, apply_rope

# hyperparameters
batch_size = 4 # independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 500
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 768
n_head = 12
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('/home/cmi_10101/Documents/coding/pytorch/architecture-implementation/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# #tokenization on the basis of character

# chars = sorted(list(set(text))) #list of characters
# vocab_size = len(chars)
# stoi = {ch:i for i, ch in enumerate(chars)}
# itos = {i:ch for i, ch in enumerate(chars)}
# encode  = lambda s: [stoi[c] for c in s] #takes a string s and output a list of numbers
# decode = lambda lst : ''.join([itos[l] for l in lst]) #decoder takes a list of numbers and outputs a joined string

# ===== Tokenizer =====
from transformers import AutoTokenizer

# HuggingFace tokenizer
use_hf_tokenizer = False   # toggle this

if use_hf_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="...")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = len(tokenizer.get_vocab())

    def encode(s):
        return tokenizer.encode(s, truncation=True, max_length=block_size)
    def decode(l):
        return tokenizer.decode(l)

else:
    # Fallback: char-level (your original code)
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

@torch.no_grad()
def estimate_loss():
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


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.mtheta_complex = compute_freq(dim=n_embd, seq_len=block_size, theta_0=10000)
        self.dropout = nn.Dropout(0.2)

    def reset_cache(self):
            self.k_cache = None
            self.v_cache = None 
            self.cache_idx = 0




    def forward(self,x, use_cache=False):
        B, T, C = x.shape #batch, time-step, n_embd (channels)
        #B, T, C -> x.shape

        # RoPE in attention with q and k 
        # RoPE(xq, xk) 
        
        k = self.key(x)  #(B, T, head_size)
        q = self.query(x) #(B, T, head_size) 
        v = self.value(x)  #B, T, head_size

        q, k = apply_rope(q, k, self.mtheta_complex)
        # k, q -> T, B, C 

        k = k.permute(1,0, 2).contiguous() #B, T, C
        q = q.permute(1, 0, 2).contiguous()  # B , T, C



        if use_cache and not self.training:
            #initialize k/v cache if empty 
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = torch.zeros(B, block_size, self.head_size, device=q.device) #b, T, HEAD_SIZE
                self.v_cache = torch.zeros(B, block_size, self.head_size, device=v.device) #B, T, HEAD_SIZE
                self.cache_idx = 0 

            #update k/v cache 
            if self.cache_idx + T <= block_size:
                self.k_cache[:, self.cache_idx:self.cache_idx+T, :] = k #passing one token at a time for inference stage 
                self.v_cache[:, self.cache_idx:self.cache_idx+T, :] = v 

                
            

            #when the cache is full it can not take the next token 
            # then we need to shift one position left 
            else:
                shift = self.cache_idx + T - block_size # shift = 1
                self.k_cache[:, :-1, :] = self.k_cache[:, 1:, :] #1 position left shift 
                self.v_cache[:, :-1, :] = self.v_cache[:, 1:, :] 

                #assign last oen to new_k and new_v
                self.k_cache[:, -T:, :] = k #here T =  1 one token by token 
                self.v_cache[:, -T:, :] = v 

                




            self.cache_idx = min(self.cache_idx+T, block_size) #should not be outbound of block size or max_seq_len 


            #calculating affinities 
            #attn q*k.T/root(2)
            wei = q @ self.k_cache.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :self.cache_idx]==0 , float("-inf"))
            wei = F.softmax(wei, dim=-1)
            
            out = wei @ self.v_cache #B, T, HS 
        else:
            # ===== TRAINING MODE (or no cache requested) =====
            # Standard full attention computation
            wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v

        # print(out.shape)
        # print("========")
        return out

class MHA(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        #O(N**2) computation 
        # self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))

        #O(N*W) computation 
        # self.heads = nn.ModuleList(LocalWindowAttention(block_size=block_size, 
        #                                                 window_size=64, 
        #                                                 causal=True) 
        #                                                 for _ in range(num_heads)
        #                                                 )

        self.head = LocalWindowAttention(
            block_size=block_size, 
            window_size=64, 
            causal=True
            )

        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def reset_cache(self):
        """Reset KV cache in all heads"""
        # for head in self.head:
        self.head.reset_cache()


    def forward(self, x, use_cache = False):
        # out = torch.cat([h(x, use_cache = use_cache) for h in self.heads], dim=-1)
        out = self.head(x, use_cache=use_cache)
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




class Feedforward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
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
        # self.ln1 = nn.LayerNorm(n_embd)
        # self.ln2 = nn.LayerNorm(n_embd)

        #replace layer norm by RMS norm

        self.norm1 = RMSNorm(dim=n_embd)
        self.norm2 = RMSNorm(dim=n_embd)
        self.use_moe = use_moe 
        if use_moe:
            self.ffn = MoEFFN(
                n_embd, n_experts, moe_loss_coeff=moe_loss_coeff
            )
        else:
            self.ffn = Feedforward(n_embd)

    def reset_cache(self):
        """Reset KV cache in attention"""
        self.sa.reset_cache()


    def forward(self, x, use_cache = False):
        # print(x.shape)
        # print()
        x = x+self.sa(self.norm1(x), use_cache=use_cache)
        # print(x.shape)
        # print()


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
            block.reset_cache()
    
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
        
        x = tok_embd

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
    

    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx

    def generate(self, model, prompt , max_new_tokens=100, top_k=50, temperature=0.8, eos_token=None):
        model.eval()
        self.reset_cache()
        idx = prompt.clone()
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -1:], use_cache = True)
            logits = logits[:, -1, :] / temperature

            # numerical safety clamp
            logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(logits, dim=-1)


            topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
            # next_token = topk_idx.gather(-1, torch.multinomial(topk_probs, 1))

            # top-k filtering and re-normalization
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # multinomial sampling
            next_token_rel = torch.multinomial(topk_probs, 1)
            next_token = topk_idx.gather(-1, next_token_rel)

            idx = torch.cat((idx, next_token), dim=1)
            if eos_token is not None and next_token.item() == eos_token:
                break
        return idx



#  move to device
model = GPT()
model = model.to(device)

# correct param-count print 
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

# sys.exit()


# turn ON MoE for some blocks (e.g., every 2nd block)
#    we don't rename vars; we reuse your Block + MoEFFN as-is.
# for i, blk in enumerate(model.blocks):
#     if hasattr(blk, "use_moe") and (i % 2 == 0):  # enable MoE on even-indexed blocks
#         use_moe = True
#         # replace the plain FFN with your MoE FFN (keeps your var names)
#         blk.ffn = MoEFFN(
#             n_embd,
#             n_experts=3,             # tweak as you like
#             expansion=4,
#             dropout=dropout,
#             moe_loss_coeff=0.01,
#         ).to(device)

# # if Expert dims were defined with a wrong inner size, patch at runtime
# with torch.no_grad():
#     probe = torch.zeros(2, 5, n_embd, device=device)
#     for mod in model.modules():
#         if isinstance(mod, Expert):
#             try:
#                 _ = mod(probe)
#             except Exception:
#                 mod.net = nn.Sequential(
#                     nn.Linear(n_embd, 4 * n_embd),
#                     nn.ReLU(),
#                     nn.Linear(4 * n_embd, n_embd),
#                 ).to(device)

# re-create optimizer (since we swapped some submodules)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


#mixed preciion training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


# ===== Checkpoint saving/loading =====
import os

checkpoint_dir = './checkpoints_v2'
os.makedirs(checkpoint_dir, exist_ok=True)

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
    return checkpoint['iter']



resume_from_checkpoint = None  # Set to checkpoint path to resume, e.g., './checkpoints/model_iter_5000.pt'
start_iter = 0
if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
    start_iter = load_checkpoint(resume_from_checkpoint, model, optimizer, scaler)
    print(f"Resuming training from iteration {start_iter}")
else:
    print("Starting training from scratch")

# Training loop
best_val_loss = float('inf')


# ===== MoE-aware training loop (loss already includes aux loss from forward) =====
for iter in range(start_iter, max_iters):

    # evaluate periodically (unchanged)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint at evaluation intervals
        checkpoint_path = os.path.join(checkpoint_dir, f'model_iter_{iter}.pt')
        save_checkpoint(model, optimizer, scaler, iter, losses['val'], checkpoint_path)

        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scaler, iter, losses['val'], best_model_path)
            print(f"New best model! Val loss: {best_val_loss:.4f}")


    # batch
    xb, yb = get_batch('train')

    # forward:PT.forward already adds aux_total to CE loss
    logits, loss = model(xb, yb)

    # ======= mixed precsiion training =================
    with autocast(dtype=torch.bfloat16):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    # loss.backward()
    # small stability bonus for MoE (optional)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # optimizer.step()


# Save final model
final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
save_checkpoint(model, optimizer, scaler, max_iters, losses['val'], final_model_path)
print("\n" + "="*50)
print(f"Training complete! Best val loss: {best_val_loss:.4f}")
print(f"Final model saved to: {final_model_path}")
print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")
print("="*50 + "\n")

# Generate sample text
print("Generating sample text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
genearted_text = decode(model.generate(model, context, max_new_tokens=500)[0].tolist()) 
print("\n" + "="*50)
print("GENERATED TEXT:")
print("="*50)
print(genearted_text)
print("="*50)
