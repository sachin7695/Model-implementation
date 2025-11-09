import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
)
from torch.utils.data import Dataset
import tiktoken

# Import your existing modules
from windowed_attention import LocalWindowAttention
from rope import compute_freq, apply_rope

# =====================================================
#           CONFIGURATION CLASS
# =====================================================
class GPTConfig(PretrainedConfig):
    model_type = "custom_gpt"
    
    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        n_head=12,
        n_layer=8,
        block_size=256,
        dropout=0.2,
        sliding_window_size=64,
        use_moe=True,
        n_experts=3,
        moe_loss_coeff=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.use_moe = use_moe
        self.n_experts = n_experts
        self.moe_loss_coeff = moe_loss_coeff


# =====================================================
#    COPY YOUR EXISTING MODEL COMPONENTS HERE
# =====================================================
# (Copy all your classes: RMSNorm, GELU, Head, MHA, Expert, 
#  Top2gate, MoEFFN, Feedforward, Block)

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
    def forward(self, x):
        return 0.5*x*(1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x, 3))
        ))


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.n_embd // config.n_head
        self.context_length = config.block_size
        self.cap_cache_to_ctx = True
        self.sliding_window_size = config.sliding_window_size

        self.w_key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.w_query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.w_value = nn.Linear(config.n_embd, self.head_size, bias=False)

        self.register_buffer('mask', 
                            torch.tril(torch.ones((config.block_size, config.block_size))),
                            persistent=False)
        self.mtheta_complex = compute_freq(dim=config.n_embd, seq_len=config.block_size, theta_0=10000)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.ptr_current_pos = 0

    def reset_cache(self):
        self.k_cache, self.v_cache = None, None
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        keys_new = self.w_key(x)
        values_new = self.w_value(x)
        q = self.w_query(x)

        if use_cache and not self.training:
            old_len = 0 if self.k_cache is None else self.k_cache.shape[1]
            if self.k_cache is None:
                self.k_cache, self.v_cache = keys_new, values_new
            else:
                self.k_cache = torch.cat([self.k_cache, keys_new], dim=1)
                self.v_cache = torch.cat([self.v_cache, values_new], dim=1)

            if self.sliding_window_size is not None:
                if self.k_cache.shape[1] > self.sliding_window_size:
                    self.k_cache = self.k_cache[:, -self.sliding_window_size:, :]
                    self.v_cache = self.v_cache[:, -self.sliding_window_size:, :]
            
            total_len = old_len + T
            k_len_now = self.k_cache.shape[1]
            dropped = max(0, total_len - k_len_now)
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
            keys, values = self.k_cache, self.v_cache
        else:
            keys, values = keys_new, values_new

        keys = keys.permute(0, 2, 1).contiguous()
        attn_scores = q @ keys

        num_tokens_q = q.shape[1]
        num_tokens_k = keys.shape[2]
        device = q.device

        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0

        q_positions = torch.arange(q_start, q_start+num_tokens_q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start+num_tokens_k, device=device, dtype=torch.long)

        win_size = num_tokens_k + 1 if self.sliding_window_size is None else self.sliding_window_size
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)
        mask_bool = (diff < 0) | (diff >= win_size)

        if use_cache and not self.training:
            self.ptr_current_pos += num_tokens_q
        else:
            self.ptr_current_pos = 0

        attn_scores.masked_fill_(mask_bool, float("-inf"))
        scale = math.sqrt(self.head_size)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ values

        return out


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def reset_cache(self):
        for head in self.heads:
            head.reset_cache()

    def forward(self, x, use_cache=False):
        out = torch.cat([h(x, use_cache=use_cache) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Expert(nn.Module):
    def __init__(self, n_embd, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion*n_embd),
            nn.ReLU(),
            nn.Linear(expansion*n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class Top2gate(nn.Module):
    def __init__(self, n_embd, n_experts, moe_loss_coeff=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(n_embd, self.n_experts, bias=False)
        self.moe_loss_coeff = moe_loss_coeff

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_2_vals, top_2_idx = probs.topk(k=2, dim=-1)

        denom = top_2_vals.sum(dim=-1, keepdim=True)
        gate_probs_top_2 = top_2_vals / denom

        importance = probs.mean(dim=(0, 1))
        top1_choice_expert_idx = probs.argmax(dim=-1)
        one_hot = F.one_hot(top1_choice_expert_idx, num_classes=self.n_experts).float()
        load = one_hot.mean(dim=(0, 1))
        aux_loss = self.n_experts * torch.sum(importance * load) * self.moe_loss_coeff
        
        return gate_probs_top_2, top_2_idx, aux_loss


class MoEFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.experts = nn.ModuleList([Expert(config.n_embd, expansion=4) for _ in range(self.n_experts)])
        self.router = Top2gate(config.n_embd, n_experts=self.n_experts, moe_loss_coeff=config.moe_loss_coeff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate_probs_top_2, top_2_idx, aux_loss = self.router(x)
        
        all_outs = torch.stack([e(x) for e in self.experts], dim=-1)
        
        B, T, _ = x.shape
        E = self.n_experts
        mask = torch.zeros(size=(B, T, E), device=x.device, dtype=x.dtype)
        mask.scatter_(dim=-1, index=top_2_idx, src=gate_probs_top_2)
        y = (all_outs * mask.unsqueeze(-2)).sum(dim=-1)
        y = self.dropout(y)

        return y, aux_loss


class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            GELU(),
            nn.Linear(config.n_embd*4, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config, use_moe=False):
        super().__init__()
        self.sa = MHA(config)
        self.norm1 = RMSNorm(dim=config.n_embd)
        self.norm2 = RMSNorm(dim=config.n_embd)
        self.use_moe = use_moe

        if use_moe:
            self.ffn = MoEFFN(config)
        else:
            self.ffn = Feedforward(config)

    def forward(self, x, use_cache=False):
        x = x + self.sa(self.norm1(x), use_cache=use_cache)

        if self.use_moe:
            y, aux = self.ffn(self.norm2(x))
            x = x + y
        else:
            y = self.ffn(self.norm2(x))
            aux = torch.tensor(0.0, device=x.device)
            x = x + y
        
        return x, aux


# =====================================================
#     HUGGING FACE COMPATIBLE MODEL
# =====================================================
class GPTForCausalLM(PreTrainedModel):
    config_class = GPTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.tok_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        self.blocks = nn.ModuleList([
            Block(config, use_moe=(i % 2 == 0))
            for i in range(config.n_layer)
        ])
        
        self.rms_norm = RMSNorm(dim=config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        self.apply(self._init_weights)
        self.current_pos = 0

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        for block in self.blocks:
            block.sa.reset_cache()
        self.current_pos = 0

    def forward(
        self,
        input_ids=None,
        labels=None,
        use_cache=False,
        return_dict=True,
        **kwargs
    ):
        """
        Forward pass compatible with HuggingFace Trainer
        
        Args:
            input_ids: Input token IDs (B, T)
            labels: Target token IDs for loss computation (B, T)
            use_cache: Whether to use KV cache
            return_dict: Whether to return ModelOutput object
        """
        B, T = input_ids.shape
        aux_total = torch.tensor(0.0, device=input_ids.device)

        # Token embeddings
        tok_embd = self.tok_embedding_table(input_ids)
        tok_embd = self.rms_norm(tok_embd)

        # Position embeddings
        if use_cache:
            pos_ids = torch.arange(
                self.current_pos, 
                self.current_pos + T, 
                device=input_ids.device, 
                dtype=torch.long
            )
            self.current_pos += T
        else:
            pos_ids = torch.arange(0, T, device=input_ids.device, dtype=torch.long)
        
        pos_embd = self.pos_embedding_table(pos_ids).unsqueeze(0)
        x = tok_embd + pos_embd

        # Transformer blocks
        for block in self.blocks:
            x, aux = block(x, use_cache=use_cache)
            aux_total += aux

        # Output layer
        x = self.rms_norm(x)
        logits = self.lm_head(x)

        # Compute loss
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss_lm = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss_lm + aux_total

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        # Return dict format (recommended for HF Trainer)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# =====================================================
#           DATASET CLASS
# =====================================================
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        attention_mask = torch.ones_like(x, dtype=torch.long)
        return {
            'input_ids': x,
            'labels': y,
            'attention_mask':attention_mask
        }


# =====================================================
#           DATA COLLATOR
# =====================================================
def collate_fn(batch):
    """
    Pads variable-length examples in a batch
    """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    masks = [item['attention_mask'] for item in batch]

    # pad sequences to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 ignores in loss
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': masks
    }


# =====================================================
#           MAIN TRAINING FUNCTION
# =====================================================
def main():
    # ===== Load and prepare data =====
    with open('/home/cmi_10101/Documents/coding/pytorch/architecture-implementation/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenization
    use_tiktoken = False
    if use_tiktoken:
        tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = 50257
        encode = lambda s: tokenizer.encode(s)
    else:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # ===== Create config and model =====
    config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=768,
        n_head=12,
        n_layer=8,
        block_size=256,
        dropout=0.2,
        sliding_window_size=64,
        use_moe=True,
        n_experts=3,
        moe_loss_coeff=0.01
    )
    
    model = GPTForCausalLM(config)
    
    print(f"\nModel has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # ===== Create datasets =====
    train_dataset = TextDataset(train_data, config.block_size)
    val_dataset = TextDataset(val_data, config.block_size)

    # ===== Training Arguments =====
    training_args = TrainingArguments(
        output_dir="./checkpoints_hf",
        overwrite_output_dir=True,
        
        # Training hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        
        # Logging
        logging_dir="./logs",
        logging_steps=50,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        
        # Performance
        gradient_accumulation_steps=1,
        fp16=False,  # Set to True if you have compatible GPU
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        dataloader_num_workers=0,
        
        # Optimization
        max_grad_norm=1.0,
        optim="adamw_torch",
        
        # Misc
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        seed=1337,
    )

    # ===== Initialize Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # ===== Train =====
    print("\n" + "="*50)
    print("Starting training with HuggingFace Trainer...")
    print("="*50 + "\n")
    
    trainer.train()

    # ===== Save final model =====
    trainer.save_model("./final_model_hf")
    print("\n✅ Training complete! Model saved to ./final_model_hf")

    # ===== Generate sample text =====
    print("\n" + "="*50)
    print("Generating sample text...")
    print("="*50)
    
    model.eval()
    start_context = "hello, i am"
    encoded = encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        # Simple generation
        for _ in range(100):
            logits = model(input_ids=encoded_tensor).logits
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            encoded_tensor = torch.cat([encoded_tensor, next_token], dim=1)
    
    # Decode (you'll need to implement decode based on your tokenizer)
    print(f"\nGenerated tokens: {encoded_tensor[0].tolist()}")
    print("="*50)


if __name__ == "__main__":
    main()