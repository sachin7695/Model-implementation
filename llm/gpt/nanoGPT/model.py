import math 
import inspect 
from dataclasses import dataclass 

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
            

        
    def forward(self, x):
        B, T, C = x.size()  # T-> sequence length C = n_embed 

        # for each B calculate q, k, v 
        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2) 
        # B, T, C -> B, T, nh, C -> B, nh, T, C
        k = k.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous() 
        q = q.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T,self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()

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







    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__() 
        self.ln_1 = nn.LayerNorm(config.n_embed) 
        self.attn = CausalSelfAttention(config) 
        self.ln_2 = nn.LayerNorm(config.n_embed) 
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
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
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    




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
    
    def forward(self, idx, targets = None):

        device = idx.device 
        b, t = idx.size()  # this t can not be more than block size         
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) 

        # forward propagation 
        tok_emb = self.transformer.wte(idx) #b, t, n_embed 
        pos_emb = self.transformer.wpe(pos) #t, n_embed 
        x = self.transformer.drop(tok_emb+pos_emb) # implicit broadcasting 
        for block in self.transformer.h:
            x = block(x) # n_layers of transformer block 

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
    

num_return_sequences = 5 
max_length = 30 
device = "cuda"

model = GPT.from_pretrained("gpt2")
print("did not crash yay!!")

model.to(device=device)
model.eval() 

#test sample 
import tiktoken 
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model, ")
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









