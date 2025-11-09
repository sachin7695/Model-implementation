import torch 
import torch.nn as nn
import math 
import torch.functional as F 
from rope import compute_freq, apply_rope

class LocalWindowAttention(nn.Module):
    def __init__(self,
                 block_size, 
                 window_size = 512,
                 embedding_dim = 768,
                 num_attention_heads = 12,
                 causal = False,
                 look_backward = 1,
                 look_forward=1,
                 attention_dropout = 0.0
                ):
        

        super(LocalWindowAttention, self).__init__()
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward 
        self.window_size = window_size
        self.embed_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads 
        self.block_size = block_size
        self.mtheta_complex = compute_freq(dim=self.head_dim, seq_len=self.block_size, theta_0=10000)

        # Projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(attention_dropout)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


        #kv cache initialization 
        self.k_cache = None 
        self.v_cache = None 
        self.cache_idx = 0 #how many tokens have been cached so far 

    def reset_cache(self):
        self.k_cache = None 
        self.v_cache = None 
        self.cache_idx = 0 

    


    def split_into_windows(self, x):
        if x.dim()==4:
            b, h, seq_len, head_dim = x.shape
            num_windows = seq_len // self.window_size
            x = x.view(b, h, num_windows, self.window_size, head_dim) 
            return x
    
        b, seq_len, d = x.shape 
        num_windows = seq_len // self.window_size 
        x = x.view(b, num_windows, self.window_size, d) 
        return x 
    
    
    def collect_windows(self, x, backward =1, forward = 1, pad_value = -1):
        if x.dim() == 4:
            batch_heads, num_windows, window_size, embed_dim = x.shape 
            pad = (0, 0, 0, 0, backward, forward)

        elif x.dim()==3:
            bath_head_dim, num_windows, window_size = x.shape 

            pad = (0, 0, backward, forward)

        x = nn.functional.pad(x, pad=pad, value=pad_value)
        gathered = []

        for i in range(num_windows): #2 windows
            start_idx = i 
            end_idx = i+forward+backward 

            grabbed_win = x[:, start_idx:end_idx+1] #[[pad], W0, W1] bs, win=3, window_size, embed_dim
            grabbed_win = grabbed_win.flatten(1, 2).unsqueeze(1) #bs, 1, 3*window_size, embed_dim

            gathered.append(grabbed_win) 
        gathered = torch.cat(gathered, dim=1) #bs, num_win, 3*win_size, embed_dim 
        ''' 
        earlier we were having for each window window_size tokens 
        but for k and v it should find the affinity to backward and forward window
        then the net length of each window should incvrease from window_size to 
        window_size*(backward+forward+1=3)
        the again make it for each window the context length to 3*window_size 
        for such num_windows 
        
        '''
        # print(gathered.shape)
        # if gathered.shape[2] == 3*self.window_size:
        #     print("yes====")
        # print("==============")
        return gathered 



    def forward_train(self, x, attention_mask = None):
        # Route to appropriate implementation

        b, ori_seq_len, d = x.shape 
        h = self.num_heads 
        head_dim = self.head_dim 

        q = self.q_proj(x).view(b, ori_seq_len, h, head_dim).permute(0, 2, 1, 3).contiguous() # b,h, ori_seq_len, head_dim 
        k = self.k_proj(x).view(b, ori_seq_len, h, head_dim).permute(0,2,1,3).contiguous()
        v = self.v_proj(x).view(b, ori_seq_len, h, head_dim).permute(0,2,1,3).contiguous()

        ### Merge together Head/Batch Dimension ###
        q = q.flatten(0,1) #batched_head(b*h), seq_len, head_dim
        k = k.flatten(0,1)
        v = v.flatten(0,1)

        q, k = apply_rope(q, k, self.mtheta_complex) 
        k = k.permute(1,0, 2).contiguous() #B*h, ori_seq_len, head_dim
        q = q.permute(1, 0, 2).contiguous()  #B*h, ori_seq_len, head_dim

        device = q.device


        if attention_mask is not None:
            attention_mask = attention_mask.repeat(self.num_heads, 1) 

        if ori_seq_len % self.window_size == 0:
            difference = self.window_size*math.ceil(ori_seq_len/self.window_size) - ori_seq_len 
            q = nn.functional.pad(q, pad=(0,0,0, difference))
            k = nn.functional.pad(k, pad=(0,0,0, difference))
            v = nn.functional.pad(v, pad=(0,0,0, difference))

        seq_len = q.shape[1]
        num_windows = seq_len // self.window_size 
        idx = torch.arange(seq_len, device=device)
        bucketed_idx = idx.reshape(1, num_windows, self.window_size) #bs,n, win_size of indexes


        ### Bucket our Q,K,V into the Chunked Windows ###
        bucketed_q = q.reshape(b*self.num_heads, num_windows, self.window_size, head_dim)
        bucketed_k = k.reshape(b*self.num_heads, num_windows, self.window_size, head_dim)
        bucketed_v = v.reshape(b*self.num_heads, num_windows, self.window_size, head_dim)


        #b*h, num_windows, 3*win_size, embed_dim
        bucketed_k = self.collect_windows(bucketed_k, self.look_backward, self.look_forward)
        bucketed_v = self.collect_windows(bucketed_v, self.look_backward, self.look_forward)

        # to know the pad token lets collect through collect window 
        # batchsize*heads, num_win, 3*win_size = 1536
        collected_bucket_idx = self.collect_windows(bucketed_idx,self.look_backward, self.look_forward)
        bucket_pad_mask = (collected_bucket_idx == -1) #b, num_win, 3*win_size 


        attention_scores = bucketed_q @ bucketed_k.transpose(-1, -2) 
        # b*h, num_win, win_size, embed_dim @ b*h, num_win, 3*win_size, embed_dim
        # attention_score has dim b*h, num_win, win_size(query tokens list length), 3*win_size(key tokens length list) 
        
        # b=1, num_win, win_size=512, 3*win_size=1536 same as attetion scores
        bucket_pad_mask = bucket_pad_mask.unsqueeze(-2).repeat(1,1,self.window_size, 1)
        ''' 
        bucket_pad_mask originally told us:
        In this window, which key tokens are fake pads?
        But attention is computed query × key.
        So we need:
        For every query token in this window, which key tokens are fake pads?
        
        '''
        attention_scores = attention_scores.masked_fill(bucket_pad_mask, float("-inf"))#b*h, num_win, 512, 1536

        #now we have to repate this for each query as its sghared across each query so add a new dim and repeat across that dim 
        #b*h, num_windows, win_size=512, 3*win_size=1536
        collected_bucket_idx = collected_bucket_idx.unsqueeze(-2).repeat(1,1,self.window_size, 1)

        # causal masking 



        #non causal masking
        if not self.causal:
            num_concat_windows = (self.look_backward + self.look_forward + 1)
            ### This is what the repeated_query_index looks like
            # tensor([[[[ 0,  0,  0,  0,  0,  0,  0,  0,  0],
            #           [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
            #           [ 2,  2,  2,  2,  2,  2,  2,  2,  2]],
            
            #          [[ 3,  3,  3,  3,  3,  3,  3,  3,  3],
            #           [ 4,  4,  4,  4,  4,  4,  4,  4,  4],
            #           [ 5,  5,  5,  5,  5,  5,  5,  5,  5]],
            
            #          [[ 6,  6,  6,  6,  6,  6,  6,  6,  6],
            #           [ 7,  7,  7,  7,  7,  7,  7,  7,  7],
            #           [ 8,  8,  8,  8,  8,  8,  8,  8,  8]],
            
            #          [[ 9,  9,  9,  9,  9,  9,  9,  9,  9],
            #           [10, 10, 10, 10, 10, 10, 10, 10, 10],
            #           [11, 11, 11, 11, 11, 11, 11, 11, 11]]]])
            repeated_query_idx = idx.reshape(1, -1, self.window_size, 1).repeat(1,1,1,self.window_size*num_concat_windows) #1, 1, win_size(512), 3*win_size(1536)

            total_look_backward = (self.window_size*self.look_backward)
            total_look_forward = (self.window_size*self.look_backward)

            max_idx = repeated_query_idx + total_look_forward 
            min_idx = repeated_query_idx - total_look_backward 

            upper_mask = ((collected_bucket_idx>max_idx) & (collected_bucket_idx !=-1))
            lower_mask = ((collected_bucket_idx<min_idx) & (collected_bucket_idx != -1))

            overcompute_mask = upper_mask | lower_mask 
            causal_mask = torch.zeros_like(attention_scores, device=device).bool() 

        else:
            num_concat_windows = (self.look_backward + self.look_backward + 1)
            repeated_query_idx = idx.reshape(1,-1,self.window_size,1).repeat(1,1,1,self.window_size*num_concat_windows)

            causal_mask = (collected_bucket_idx>repeated_query_idx)
            total_look_backward = (self.window_size*self.look_backward)
            min_idx = repeated_query_idx - total_look_backward 
            overcompute_mask = ((collected_bucket_idx<min_idx) & (collected_bucket_idx != -1))

        if attention_mask is not None:

            ''' 
            this attention mask is for making the sequence_len uniform from 
            data collator and dataloader while data preparation for input 
            after tokenization step 
            we dont want query to attend those tokens so we mask out them and 
            dont involve them in attention calculation
            this is done with padding 

            apart form another padding being to make the sequence_len % num_windows == 0 
            '''

            if ori_seq_len % self.window_size !=0 :
                diff = self.window_size * math.ceil(ori_seq_len/self.window_size) - ori_seq_len
                attention_mask = nn.functional.pad(attention_mask, pad=(0, diff), value=-1)

            #chunk into buckets 
            # b*h, 4, 3
            attention_mask = attention_mask.reshape(b*h, num_windows, self.window_size)
            #includes mask to neighbours (3*win_size)
            #b*h, num_windows, 3*win_size
            attention_mask = self.collect_windows(attention_mask, self.look_backward, self.look_forward)
            #b*h, num_windows, 3*win_size -> b*h, num_windows, win_size, 3*win_Size
            ''' 
            Before: (batch_heads, num_windows, key_len)
            After : (batch_heads, num_windows, query_len, key_len)
            '''

            attention_mask = attention_mask.unsqueeze(-2).repeat(1,1,self.window_size,1)
            attention_mask = (attention_mask==0) 

        else:
            # we want every query token to attend the other token no padded token in the sequence
            # 1 means masked out 0  means dont maks out that token or index 
            attention_mask = torch.zeros_like(attention_scores, device=device).bool()


        combined_mask = overcompute_mask | attention_mask 
        if self.causal:
            combined_mask = combined_mask | causal_mask 

        #replace the masked value with -inf for softmax
        mask_value = float("-inf")
        attention_scores = attention_scores.masked_fill(combined_mask, mask_value)

        #softmax 
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.dropout(attention_scores) 
        #b*h, num_windows, query_len(win_size), key_len (3*win_size)

        # attn @ v
        #b*h, num_win, win_size, 3*win_size @ b*h, num_win, 3*win_size, head_dim
        # output b*h, num_win, win_size, head_dim
        output = attention_scores @ bucketed_v 
        output = output.reshape(b, h, -1, head_dim) #b, h, num_win*win_size, head_dim
        output = output[:, :, :ori_seq_len].permute(0,2,1,3).flatten(2)  #b, orig_seq_len, embed_dim 
        output = self.out_proj(output)


        return output 
    

    # ===========  My naive implementation of KV-cache for sliding window attention during inference forward propagation  =============
    
    def forward_inference(self, x, attention_mask = None):

        B, T, C = x.shape
        h = self.num_heads 
        head_dim = self.head_dim 
        device = x.device 

        #project only new token 
        # q (b, h, 1, head_dim)
        # k(b, h, 1, head_dim) 
        # v(b, h, 1, head_dim) 

        # Project new tokens
        q_new = self.q_proj(x).view(B, T, h, head_dim).permute(0, 2, 1, 3).contiguous()  # (B, h, T, head_dim)
        k_new = self.k_proj(x).view(B, T, h, head_dim).permute(0, 2, 1, 3).contiguous()
        v_new = self.v_proj(x).view(B, T, h, head_dim).permute(0, 2, 1, 3).contiguous()

        # Initialize cache if needed
        if self.k_cache is None or self.v_cache is None:
            max_cache_len = self.block_size
            self.k_cache = torch.zeros(B, h, max_cache_len, head_dim, device=device)
            self.v_cache = torch.zeros(B, h, max_cache_len, head_dim, device=device)
            self.cache_idx = 0 

        cache_start = self.cache_idx 
        cache_end = self.cache_idx + T 

        # Handle cache overflow with sliding window
        if cache_end > self.k_cache.shape[2]:
            # Shift cache left
            shift_amount = cache_end - self.k_cache.shape[2]
            self.k_cache[:, :, :-shift_amount, :] = self.k_cache[:, :, shift_amount:, :].clone()
            self.v_cache[:, :, :-shift_amount, :] = self.v_cache[:, :, shift_amount:, :].clone()
            cache_start = self.k_cache.shape[2] - T
            cache_end = self.k_cache.shape[2]

        # Store new keys and values in cache
        self.k_cache[:, :, cache_start:cache_end, :] = k_new
        self.v_cache[:, :, cache_start:cache_end, :] = v_new 
        self.cache_idx = cache_end

        # Determine valid range for attention based on sliding window
        last_pos = cache_end - 1
        current_window_idx = last_pos // self.window_size 
        start_window_idx = current_window_idx * self.window_size 
        valid_start_idx = max(0, start_window_idx - self.look_backward * self.window_size)
        valid_end_idx = cache_end

        # Extract relevant keys and values from cache
        k_relevant = self.k_cache[:, :, valid_start_idx:valid_end_idx, :] # (B, h, valid_length, head_dim)
        v_relevant = self.v_cache[:, :, valid_start_idx:valid_end_idx, :]

        # print(f" shape of k_rel {k_relevant.shape} and shape of v_rel is {v_relevant.shape}")

        
        # Apply RoPE
        # For now, reshaping for attention computation
        q_attn = q_new.reshape(B * h, T, head_dim)
        k_attn = k_relevant.reshape(B * h, -1, head_dim)
        v_attn = v_relevant.reshape(B * h, -1, head_dim)

        # Compute attention scores
        attention_score = q_attn @ k_attn.transpose(-1, -2)  # (B*h, T, valid_length)
        attention_score = attention_score / math.sqrt(head_dim)

        # Create causal mask (vectorized)
        query_pos = torch.arange(cache_start, cache_end, device=device).unsqueeze(1)  # (T, 1)
        key_pos = torch.arange(valid_start_idx, valid_end_idx, device=device).unsqueeze(0)  # (1, valid_length)

        #inefficient way with loop but intuitive

        # Query at position i can only attend to keys at positions <= i
        # causal_mask = []
        # for i in range(len(key_pos)): #val_len times
        #     for j in range(len(query_pos)): #T times 
        #         if key_pos[i] <= query_pos[j]: 
        #             causal_mask.append(torch.tensor([True], device=device)) 
        #         else:
        #             causal_mask.append(torch.tensor([False], device=device)) 
        # causal_mask = torch.cat(causal_mask, dim = 0)
        # causal_mask = causal_mask.reshape(T, len(key_pos)) #T, valid range 
        # causal_mask = causal_mask.unsqueeze(0) #Batch dim added B, T, valid_range T is 1 as we are doing token by token 


        #efficient-way of making the causal masking
        causal_mask = key_pos > query_pos  # (T, valid_length)
        causal_mask = causal_mask.unsqueeze(0)  # (1, T, valid_length)

        # --- Window Boundary Mask ---
        # Ensure tokens don't attend beyond sliding window limits
        # Calculate distance between each query and key position
        distances = query_pos - key_pos  # (T, valid_length)
        max_backward = self.look_backward * self.window_size
        max_forward = self.look_forward * self.window_size


        # Mask tokens that are too far backward or forward
        window_mask = (distances > max_backward) | (distances < -max_forward)
        window_mask = window_mask.unsqueeze(0)  # (1, T, valid_length)

        # Combined mask
        combined_mask = causal_mask | window_mask if self.causal else window_mask
        attention_score = attention_score.masked_fill(combined_mask, float("-inf")) #b*h, T, valid_range 
        
        attention_probs = attention_score.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        output = attention_probs @ v_attn  # (B*h, T, head_dim)
        output = output.view(B, h, T, head_dim)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.flatten(2)  # (B, T, embed_dim)
        output = self.out_proj(output)
        
        return output
    


    def forward(self, x, attention_mask = None, use_cache = False):


        if use_cache and not self.training:
            return self.forward_inference(x, attention_mask=attention_mask)
        
        else:
            return self.forward_train(x, attention_mask=attention_mask)



# if __name__ == "__main__":

#     rand = torch.randn(4,256,768)
#     attention = LocalWindowAttention(window_size=64, causal=True)
#     out = attention(rand)
#     print(out.shape)



if __name__ == "__main__":
    print("Testing LocalWindowAttention with KV-cache")
    
    # Create model
    model = LocalWindowAttention(
        block_size=256,
        window_size=64,
        embedding_dim=768,
        num_attention_heads=12,
        causal=True
    )
    
    # Test training mode
    print("\n1. Testing training mode (full sequence):")
    model.train()
    train_input = torch.randn(2, 256, 768)
    train_output = model(train_input, use_cache=False)
    print(f"Input shape: {train_input.shape}")
    print(f"Output shape: {train_output.shape}")
    
    # Test inference mode
    print("\n2. Testing inference mode (token-by-token with cache):")
    model.eval()
    model.reset_cache()
    
    for step in range(5):
        token = torch.randn(1, 1, 768)
        output = model(token, use_cache=True)
        print(f"Step {step}: cache_idx = {model.cache_idx}, output shape = {output.shape}")
    
    print("\n All tests passed!")