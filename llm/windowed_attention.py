import torch 
import torch.nn as nn
import math 
import torch.functional as F 


class LocalWindowAttention(nn.Module):
    def __init__(self, 
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
        print(gathered.shape)
        if gathered.shape[2] == 3*self.window_size:
            print("yes====")
        print("==============")
        return gathered 



    def forward_train(self, x, attention_mask = None):
        # Route to appropriate implementation

        b, ori_seq_len, d = x.shape 
        h = self.num_heads 
        head_dim = self.head_dim 

        q = self.q_proj(x).view(b, ori_seq_len, h, head_dim).permute(0, 2, 1, 3).contiguous()
        k = self.k_proj(x).view(b, ori_seq_len, h, head_dim).permute(0,2,1,3).contiguous()
        v = self.k_proj(x).view(b, ori_seq_len, h, head_dim).permute(0,2,1,3).contiguous()

        ### Merge together Head/Batch Dimension ###
        q = q.flatten(0,1) #batched_head(b*h), seq_len, head_dim
        k = k.flatten(0,1)
        v = v.flatten(0,1)
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
        # batchsize*heads, num_win, win_size
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
        attention_scores = attention_scores.masked_fill(bucket_pad_mask, float("-inf"))
        collected_bucket_idx = collected_bucket_idx.unsqueeze(-2).repeat(1,1,self.window_size, 1)

        # causal masking 



        #non causal masking
        if not self.causal:
            num_concat_windows = (self.look_backward + self.look_forward + 1)
            repeated_query_idx = idx.reshape(1, -1, self.window_size, 1).repeat(1,1,1,self.window_size*num_concat_windows)

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
    
    def forward_inference(self, x, attention_mask = None):
        B, T, C = x.shape  #usally T = 1 
        h = self.num_heads 
        head_dim = self.head_dim 
        device = x.device 

        #project only new token 
        # q (b, h, 1, head_dim)
        # k(b, h, 1, head_dim) 
        # v(b, h, 1, head_dim) 

        q_new = self.q_proj(x).view(B, T, h, head_dim).permute(0, 2, 1, 3).contiguous()
        k_new = self.k_proj(x).view(B, T, h, head_dim).permute(0, 2, 1, 3).contiguous()
        v_new = self.v_proj(x).view(B, T, h, head_dim).permute(0, 2, 1, 3).contiguous()

        if self.k_cache is None or self.v_cache is None:
            max_cache_lean = 8192 
            self.k_cache = torch.zeros(B, h, max_cache_lean, head_dim, device=device)
            self.v_cache = torch.zeros(B, h, max_cache_lean, head_dim, device=device)
            self.cache_idx = 0 

        cache_start = self.cache_idx 
        cache_end = self.cache_idx + T 

        if cache_end > self.k_cache.shape[2]:
            # Shift cache left (like your reference)
            shift_amount = cache_end -  self.k_cache.shape[2] 
            self.k_cache[:, :, :-shift_amount, :] = self.k_cache[:, :, shift_amount:, :]
            self.v_cache[:, :, :-shift_amount, :] = self.v_cache[:, :, shift_amount:, :]

            cache_start = self.k_cache.shape[2] - T 
            cache_end = self.k_cache.shape[2]
            self.cache_idx = cache_start 

        self.k_cache[:, :, cache_start:cache_end,:] = k_new 
        self.v_cache[:, :, cache_start:cache_end, :] = v_new 

        current_pos = cache_end - 1 
        #which window this position belongs to 
        current_window_idx = current_pos // self.window_size 






    def forward(self, x, attention_mask = None, use_cache = False):


        if use_cache and not self.training:
            return self.forward_train(x, attention_mask=attention_mask)
        
        else:
            return self.forward_inference(x, attention_mask=attention_mask)

        # b, ori_seq_len, d = x.shape 
        # h = self.num_heads 
        # head_dim = self.head_dim 

        # q = self.q_proj(x).view(b, ori_seq_len, h, head_dim).permute(0, 2, 1, 3).contiguous()
        # k = self.k_proj(x).view(b, ori_seq_len, h, head_dim).permute(0,2,1,3).contiguous()
        # v = self.k_proj(x).view(b, ori_seq_len, h, head_dim).permute(0,2,1,3).contiguous()

        # ### Merge together Head/Batch Dimension ###
        # q = q.flatten(0,1) #batched_head(b*h), seq_len, head_dim
        # k = k.flatten(0,1)
        # v = v.flatten(0,1)
        # device = q.device


        # if attention_mask is not None:
        #     attention_mask = attention_mask.repeat(self.num_heads, 1) 

        # if ori_seq_len % self.window_size == 0:
        #     difference = self.window_size*math.ceil(ori_seq_len/self.window_size) - ori_seq_len 
        #     q = nn.functional.pad(q, pad=(0,0,0, difference))
        #     k = nn.functional.pad(k, pad=(0,0,0, difference))
        #     v = nn.functional.pad(v, pad=(0,0,0, difference))

        # seq_len = q.shape[1]
        # num_windows = seq_len // self.window_size 
        # idx = torch.arange(seq_len, device=device)
        # bucketed_idx = idx.reshape(1, num_windows, self.window_size) #bs,n, win_size of indexes


        # ### Bucket our Q,K,V into the Chunked Windows ###
        # bucketed_q = q.reshape(b*self.num_heads, num_windows, self.window_size, head_dim)
        # bucketed_k = k.reshape(b*self.num_heads, num_windows, self.window_size, head_dim)
        # bucketed_v = v.reshape(b*self.num_heads, num_windows, self.window_size, head_dim)


        # #b*h, num_windows, 3*win_size, embed_dim
        # bucketed_k = self.collect_windows(bucketed_k, self.look_backward, self.look_forward)
        # bucketed_v = self.collect_windows(bucketed_v, self.look_backward, self.look_forward)
        # # to know the pad token lets collect through collect window 
        # # batchsize*heads, num_win, win_size
        # collected_bucket_idx = self.collect_windows(bucketed_idx,self.look_backward, self.look_forward)
        # bucket_pad_mask = (collected_bucket_idx == -1) #b, num_win, 3*win_size 


        # attention_scores = bucketed_q @ bucketed_k.transpose(-1, -2) 
        # # b*h, num_win, win_size, embed_dim @ b*h, num_win, 3*win_size, embed_dim
        # # attention_score has dim b*h, num_win, win_size(query tokens list length), 3*win_size(key tokens length list) 
        
        # # b=1, num_win, win_size=512, 3*win_size=1536 same as attetion scores
        # bucket_pad_mask = bucket_pad_mask.unsqueeze(-2).repeat(1,1,self.window_size, 1)
        # ''' 
        # bucket_pad_mask originally told us:
        # In this window, which key tokens are fake pads?
        # But attention is computed query × key.
        # So we need:
        # For every query token in this window, which key tokens are fake pads?
        
        # '''
        # attention_scores = attention_scores.masked_fill(bucket_pad_mask, float("-inf"))
        # collected_bucket_idx = collected_bucket_idx.unsqueeze(-2).repeat(1,1,self.window_size, 1)

        # # causal masking 



        # #non causal masking
        # if not self.causal:
        #     num_concat_windows = (self.look_backward + self.look_forward + 1)
        #     repeated_query_idx = idx.reshape(1, -1, self.window_size, 1).repeat(1,1,1,self.window_size*num_concat_windows)

        #     total_look_backward = (self.window_size*self.look_backward)
        #     total_look_forward = (self.window_size*self.look_backward)

        #     max_idx = repeated_query_idx + total_look_forward 
        #     min_idx = repeated_query_idx - total_look_backward 

        #     upper_mask = ((collected_bucket_idx>max_idx) & (collected_bucket_idx !=-1))
        #     lower_mask = ((collected_bucket_idx<min_idx) & (collected_bucket_idx != -1))

        #     overcompute_mask = upper_mask | lower_mask 

        #     causal_mask = torch.zeros_like(attention_scores, device=device).bool() 

        # else:
        #     num_concat_windows = (self.look_backward + self.look_backward + 1)
        #     repeated_query_idx = idx.reshape(1,-1,self.window_size,1).repeat(1,1,1,self.window_size*num_concat_windows)

        #     causal_mask = (collected_bucket_idx>repeated_query_idx)
        #     total_look_backward = (self.window_size*self.look_backward)
        #     min_idx = repeated_query_idx - total_look_backward 
        #     overcompute_mask = ((collected_bucket_idx<min_idx) & (collected_bucket_idx != -1))

        # if attention_mask is not None:

        #     ''' 
        #     this attention mask is for making the sequence_len uniform from 
        #     data collator and dataloader while data preparation for input 
        #     after tokenization step 
        #     we dont want query to attend those tokens so we mask out them and 
        #     dont involve them in attention calculation
        #     this is done with padding 

        #     apart form another padding being to make the sequence_len % num_windows == 0 
        #     '''

        #     if ori_seq_len % self.window_size !=0 :
        #         diff = self.window_size * math.ceil(ori_seq_len/self.window_size) - ori_seq_len
        #         attention_mask = nn.functional.pad(attention_mask, pad=(0, diff), value=-1)

        #     #chunk into buckets 
        #     # b*h, 4, 3
        #     attention_mask = attention_mask.reshape(b*h, num_windows, self.window_size)
        #     #includes mask to neighbours (3*win_size)
        #     #b*h, num_windows, 3*win_size
        #     attention_mask = self.collect_windows(attention_mask, self.look_backward, self.look_forward)
        #     #b*h, num_windows, 3*win_size -> b*h, num_windows, win_size, 3*win_Size
        #     ''' 
        #     Before: (batch_heads, num_windows, key_len)
        #     After : (batch_heads, num_windows, query_len, key_len)
        #     '''

        #     attention_mask = attention_mask.unsqueeze(-2).repeat(1,1,self.window_size,1)
        #     attention_mask = (attention_mask==0) 

        # else:
        #     # we want every query token to attend the other token no padded token in the sequence
        #     # 1 means masked out 0  means dont maks out that token or index 
        #     attention_mask = torch.zeros_like(attention_scores, device=device).bool()


        # combined_mask = overcompute_mask | attention_mask 
        # if self.causal:
        #     combined_mask = combined_mask | causal_mask 

        # #replace the masked value with -inf for softmax
        # mask_value = float("-inf")
        # attention_scores = attention_scores.masked_fill(combined_mask, mask_value)

        # #softmax 
        # attention_scores = attention_scores.softmax(dim=-1)
        # attention_scores = self.dropout(attention_scores) 
        # #b*h, num_windows, query_len(win_size), key_len (3*win_size)

        # # attn @ v
        # #b*h, num_win, win_size, 3*win_size @ b*h, num_win, 3*win_size, head_dim
        # # output b*h, num_win, win_size, head_dim
        # output = attention_scores @ bucketed_v 
        # output = output.reshape(b, h, -1, head_dim) #b, h, num_win*win_size, head_dim
        # output = output[:, :, :ori_seq_len].permute(0,2,1,3).flatten(2)  #b, orig_seq_len, embed_dim 
        # output = self.out_proj(output)


        # return output


        # q = self.split_into_windows(q) #b, h, num_w, win_size,head_dim
        # k = self.split_into_windows(k)
        # v = self.split_into_windows(v) 

        # #attention in each windows 
        # scores = q @ k.transpose(-1, -2) #b, h, num_w, win_size, win_size 
        # scores = scores/math.sqrt(head_dim) 
        # attn = torch.softmax(scores, dim=-1)
        # out = attn @ v #b, h, num_w, win_size, head_dim 

        # out = out.reshape(b, h, seq_len, head_dim).permute(0,2,1,3).reshape(b, seq_len, d)
        # return scores, self.out_proj(out)


if __name__ == "__main__":

    rand = torch.randn(4,256,768)
    attention = LocalWindowAttention(window_size=64, causal=True)
    out = attention(rand)
    print(out.shape)