import math 
import inspect , time
from dataclasses import dataclass 
import sys
import torch 
import torch.nn as nn 
from torch.nn import functional as F  


@dataclass 
class GPTConfig:
    block_size: int = 1024 
    vocab_size: int = 50257 
    n_layer: int = 12 
    n_head: int = 12
    n_embed: int = 768 
    bias: bool = False
    dropout: float = 0.2


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__() 
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed) 
        self.gelu = nn.GELU(approximate="tanh") 
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed) 

    def forward(self, x):
        x = self.c_fc(x) 
        x = self.gelu(x) 
        x = self.c_proj(x) 
        return x 
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__() 
        assert config.n_embed % config.n_head == 0

        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed, bias = config.bias) # split into 3  
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias) 
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout) 
        self.n_head = config.n_head 
        self.n_embed = config.n_embed 
        self.dropout = config.dropout 
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
        if not self.flash:
            print("Using slow attention!!!")
            self.register_buffer("bias", torch.tril(torch.ones(size=(config.block_size, config.block_size))
                                                    .view(1, 1, config.block_size, config.block_size)))
            
        # self.c_proj.NANOGPT_SCALE_INIT = 1
        # KV cache buffer
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        
    def forward(self, x, use_cache = False):
        B, T, C = x.size()  # T-> sequence length C = n_embed 

        # for each B calculate q, k, v
        k_new, v_new, q = self.c_attn(x).split(self.n_embed, dim=2)
        k_new =  k_new.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        q = q.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        v_new = v_new.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()


        ####### KV CACHE IMPLEMENTATION #######
        if use_cache and not self.training:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new 
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=1)
            k, v = self.cache_k, self.cache_v 
        else:
            k, v = k_new, v_new 
        
        ####### OLD IMPLEMENTATION #######
        # q, k, v = self.c_attn(x).split(self.n_embed, dim = 2) 
        # B, T, C -> B, T, nh, C -> B, nh, T, C
        # k = k.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous() 
        # q = q.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        # v = v.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q,k, v, attn_mask=None, 
                                                                dropout_p=self.dropout if self.training else 0,
                                                                is_causal=True)
        else:
            # manual implementation of attention 
            k_transpose = k.permute(0, 1, 3, 2).contiguous() # B, nh, T, C -> B, nh, C, T
            attn = (q @ k_transpose)*(1/math.sqrt(k.shape[-1])) # B, nh, T, T
            attn = attn.masked_fill(self.bias[:, :, :T, :T]==0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn) 
            y = attn @ v # B, nh, T, T @ B, nh, T, C -> B, nh, T, C 
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y  


    def reset_cache(self):
            self.cache_k, self.cache_v = None, None 
            self.ptr_curr_pos = 0



    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__() 
        self.ln_1 = nn.LayerNorm(config.n_embed) 
        self.attn = CausalSelfAttention(config) 
        self.ln_2 = nn.LayerNorm(config.n_embed) 
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x 
    




class GPT(nn.Module):

    def __init__(self, config):
        super(GPT, self).__init__() 
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        self.current_pos = 0

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # init params 
        self.apply(self._init_weights)

        # To solve the growing variance 
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)) # per attention 2 resdiual

    

    # init weights 
    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    @classmethod 
    def from_pretrained(cls, model_type):
        ''' loads pretariend gpt2 model from huggingface '''
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel 
        print("loading weights from pretarined gpt2: %s" % model_type)


        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints

        # create a from-scartch minigpt model 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() 
        sd_keys = sd.keys() 
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param


        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model 
    
    def forward(self, idx, targets = None, use_cache = False):

        device = idx.device 
        b, t = idx.size()  # this t can not be more than block size         
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) 

        if use_cache:
            pos = torch.arange(self.current_pos, self.current_pos+t, device=idx.device, dtype=torch.long)
            self.current_pos += t 
        else:
            pos = torch.arange(0, t, device=idx.device, dtype=torch.long)
    
        pos_emb = self.transformer.wpe(pos).unsqueeze(0)

        # forward propagation 
        tok_emb = self.transformer.wte(idx) #b, t, n_embed 
        # pos_emb = self.transformer.wpe(pos) #t, n_embed 

        x = self.transformer.drop(tok_emb+pos_emb) # implicit broadcasting 
        for block in self.transformer.h:
            x = block(x, use_cache=use_cache) # n_layers of transformer block 

        x = self.transformer.ln_f(x) # layer normalization 
        loss = None 
        if targets is not None:
            logits  = self.lm_head(x) #b, t, vocab_size =50257
            logits = logits.view(b*t, logits.shape[-1]) #b*t, 50257
            # logits = logits[:, -1, :] # shape b, 50257
            loss = F.cross_entropy(logits, targets.view(-1), ignore_index=-1) # targets.shape b*t (N, )
        else:
            # inference time only pass the last time step 
            logits = self.lm_head(x)
            # logits = logits[:, -1, :] # last time step 
        return logits, loss
    
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0

    

    def generate(self, model, idx, max_new_tokens, temperature = 1.0, top_k = None):
        model.eval()
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else \
            idx[:, -self.config.block_size] # take the block_size window size context 

            logits, loss = model(idx) 
            logits = logits[:, -1, :]/temperature 
            # top_k probs 
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
    
    def generate_text_simple_cached(self,model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
        model.eval()
        ctx_len = context_size or model.pos_emb.num_embeddings

        with torch.no_grad():
            if use_cache:
                # Init cache with full prompt
                model.reset_kv_cache()
                logits = model(idx[:, -ctx_len:], use_cache=True)

                for _ in range(max_new_tokens):
                    # a) pick the token with the highest log-probability (greedy sampling)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    # b) append it to the running sequence
                    idx = torch.cat([idx, next_idx], dim=1)
                    # c) feed model only the new token
                    logits = model(next_idx, use_cache=True)
            else:
                for _ in range(max_new_tokens):
                    logits = model(idx[:, -ctx_len:], use_cache=False)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    idx = torch.cat([idx, next_idx], dim=1)

        return idx
    

num_return_sequences = 6
max_length = 50 
device = "cuda"

import tiktoken 

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B 
        self.T = T 

        with open("/home/cmi_10101/Documents/coding/pytorch/architecture-implementation/input.txt", mode="r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text) 
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches") 

        # state
        self.current_position = 0 

    def next_batch(self):
        B,T = self.B, self.T 
        buf = self.tokens[self.current_position:self.current_position+B*T+1] #exact B*T times tokens
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position+=B*T 
        # loading the next batch reset the self.current_position
        '''
        That means we’ve hit the end of the dataset, 
        so the next call will start reading from the beginning again (fresh epoch).
        '''
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0 
        x, y = x.to(device), y.to(device)

        return x, y 

# learning rate decay scheduler (cosine with warmup)
max_steps = 50
max_lr = 6e-4 
min_lr = 6e-4 * 0.1 # 10 percent of max_lr 
warmup_steps = 10
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / (warmup_steps + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
train_loader = DataLoaderLite(B = 8, T = 64)
# earlier it was FP 32 right now its is TF32 (8x) throughput improvement
torch.set_float32_matmul_precision('high') # not all gpu has TF32


# get logits 
model = GPT(GPTConfig(vocab_size=50304)) # 50304 is a power of 2
model.to(device)
# compile the Model
model = torch.compile(model) 

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps):
    t0 = time.time()
    x,y = train_loader.next_batch() # x, y already in GPU!!
    optimizer.zero_grad()
    # Autocast mixed precision training limited to logits and loss calculation 
    # rest of the operations are TF32
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()

    # clipping the gradient norm
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) 
    # Determine and set the lr for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    torch.cuda.synchronize() # wait till the task done from gpu
    t1 = time.time()
    dt = (t1-t0)*1000 # in ms 
    tokens_per_second = (train_loader.B*train_loader.T)/(t1-t0)
    print(f"step {step}, loss: {loss.item()} | norm {norm:.4f} | dt: {dt:.2f}ms, tok/sec: {tokens_per_second:.2f}")



sys.exit(0)







model = GPT.from_pretrained("gpt2")
# print("did not crash yay!!")

model.to(device=device)
model.eval() 

tokens = torch.tensor(tokens, dtype=torch.long, device=device) #(t,)
#(t, ) -> (5, t)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeat across first dim
x = tokens.to(device=device)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x) 
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        idx_next = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, dim=-1, index=idx_next)
        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist() 
    decoded = enc.decode(tokens)
    print(">", decoded)









