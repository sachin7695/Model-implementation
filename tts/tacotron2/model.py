import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
from dataclasses import dataclass 


@dataclass
class Tacotron2Config:

    # Mel Input Features #
    num_mels: int = 80 

    # Character Embeddings #
    character_embed_dim: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    # Encoder config #
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5
    
    # Decoder Config #
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout_p: float = 0.5
    decoder_dropout_p: float = 0.1

    # Attention Config #
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout_p: float = 0.1 



class LinearNorm(nn.Module):
    # standard Linear layer with different weights init method 
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 w_init_gain = "linear"):
        super(LinearNorm, self).__init__()

        self.linear = nn.Linear(in_features=in_features,out_features=out_features, bias=bias)

        #diff weights for different activation function 
        #for linear weight gain is 1 

        '''https://docs.pytorch.org/docs/stable/nn.init.html'''
        nn.init.xavier_uniform_(
            self.linear.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear(x)
    

class ConvNorm(nn.Module):
    #standard Conv layer with diff weight init method 

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 1,
                 stride = 1,
                 padding = None,
                 dilation=1,
                 bias = True,
                 w_init_gain = "linear"):
        super(ConvNorm, self).__init__() 

        if padding is None:
            padding = "same" #input dim == output dim no downsampling 
            #as we want to keep the sequence length intact for lstm 

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        nn.init.xavier_uniform(
            self.conv.weight, gain = nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    # input text is being processed embedding layer -> conv -> lstm 
    def __init__(self, config):
        super(Encoder, self).__init__() 

        self.embeddings = nn.Embedding(config.num_chars,
                                       embedding_dim=config.character_embed_dim,
                                       padding_idx=config.pad_token_id,
                                       )
        self.convolutions = nn.ModuleList() 

        for i in range(config.encoder_n_convolutions):

            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=config.encoder_embed_dim if i!=0 else config.character_embed_dim,
                        out_channels=config.encoder_embed_dim,
                        kernel_size=config.encoder_kernel_size,
                        stride=1,
                        padding="same",
                        dilation = 1,
                        w_init_gain="relu"
                    ),

                    nn.BatchNorm1d(config.encoder_embed_dim),
                    nn.ReLU(),
                    nn.Dropout(config.encoder_dropout_p)
                )
            )
            self.lstm = nn.LSTM(
                input_size=config.encoder_embed_dim,
                hidden_size=config.encoder_embed_dim//2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

    def forward(self, x, input_lengths = None):
        x = self.embeddings(x) #b, num_chars, embeddim_dim 
        x = x.permute(0, 2, 1).contiguous() #transpose(1,2) B, E, T  # for conv layer we did this

        batch_size, channels, seq_len = x.shape  #embedding dim  == channels

        #input_lengths batch size = 2 , [4, 5] batch 1 seq_len 4,,, and batch 2 seq_len 5
        if input_lengths is None:
            input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, device=x.device)

        for block in self.convolutions:
            x = block(x) 

        # for lstm again convert back to B, T, E 
        x = x.permute(0, 2, 1).contiguous() 

        # Pack Padded Sequence so LSTM doesnt Process Pad Tokens 
        # This requires data to be sorted in longest to shortest!! 
        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True)
    
        # Pass Data through LSTM 
        outputs, _ = self.lstm(x)

        # Pad Packed Sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )

        return outputs
    

class Prenet(nn.Module):

    # at each decoder step we pass the prev generated mel spec to 
    # to pre net layer to get the feature extracted and pass to the lstm and attention  

    def __init__(self, 
                 input_dim, # 80 mel bins 80 dim vector for each time frame
                 prenet_dim,  #256
                 prenet_depth, # 2 layer
                 dropout_p = 0.5):
        super(Prenet, self).__init__()

        self.dropout_p = dropout_p 
        dims = [input_dim] + [prenet_dim for _ in range(prenet_depth)] #[80, 256, 256] 

        #1st layer (80, 256)
        # 2nd layer (256, 256) 

        #zip the dim or pair it  
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(
                nn.Sequential(
                    LinearNorm(in_features=in_dim,
                               out_features=out_dim,
                               bias=False,
                               w_init_gain="relu"
                    ),
                    nn.ReLU()

                )
            )

            # zipping logic -> 1st (80, 256) -> 2nd (256, 256) -> output dim 256 dim hidden vector of acoustic feature 

    def forward(self, x): 
        #B, T, mel_bins = x.shape 
        for layer in self.layers:
            # Even during inference we leave this dropout enabled to "introduce output variation" #
            x = F.dropout(layer(x), p=self.dropout_p, training=True)
        return x #B, T , 256 
    
class LocationLayer(nn.Module):

    def __init__(self,
                 attention_n_filters,
                 attention_kernel_size,
                 attention_dim=128):
        super(LocationLayer, self).__init__()

        self.conv = ConvNorm(
            in_channels=2 , #attention weights shape (B, 2, 128) -> (B, 32, 128)
            out_channels=attention_n_filters, kernel_size=attention_kernel_size,
            padding="same",
            bias=False
        ) 


        # (B, 32, 128) -> (B, 12)
        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights):
        #Attention_weights.shape (B, 2, seq_len)  
        attention_weights = self.conv(attention_weights).permute(0, 2, 1).contiguous()
        #attention_weights.shape (B, 32, seq_len)
        attention_weights = self.proj(attention_weights) # (B,128, seq_len)
        return attention_weights

class LocalSensitiveAttention(nn.Module):

    def __init__(self, 
                 attention_dim,
                 decoder_hidden_size, # after passing through lstm cell 1024
                 encoder_hidden_size,
                 attention_n_filters,
                 attention_kernel_size):
        super(LocalSensitiveAttention, self).__init__()


        # lstm layer hidden dim to attention compressed dim  1024 -> 128
        self.in_proj = LinearNorm(decoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")

        # encoder dim to attention compressed dim 512 -> 128
        self.enc_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")


        # Convolution layer to project  (B, 2, seq_len) -> (B, 32, seq_len)  feat. vec having the alignment feat. so far
        self.what_i_have_said = LocationLayer(
            attention_n_filters,
            attention_kernel_size,
            attention_dim
        ) 



        self.energy_proj = LinearNorm(attention_dim, 1, bias=False, w_init_gain="tanh")
        self.reset() 

    def reset(self):
        self.enc_proj_cache = None 

    def _calculate_alignment_energies(self, 
                                    mel_input,  #(B, 1024), each time step 80 dim vector pass through lstm for 1024 dim
                                    encoder_output, #(B, #chars, 512)
                                    cumulative_concat_attention_weights, #(B, 2, #chars)
                                    mask=None):

        # Take our previous step of the mel sequence and project it (B x 1 x attention_dim)
        mel_proj = self.in_proj(mel_input) # (B, 1024) -> (B, attention_dim=128)
        mel_proj = mel_proj.unsqueeze(1) # (B x 1 x attention_dim=128)

        # Take our entire encoder output and project it to (B x #chars x attention_dim)
        # encoder_len means number of characters that text has 
        # you need to compute only once as the number of characters is constant for all time steps 
        # that's why we can put it in a cache  to reuse for remaining time steps

        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output) #(B, #chars, 128)

        # Look at our attention weight history to understand where the model has already placed attention
        # (B, 2, #chars) -> (B, 32, #chars) -> (B, #chars, 128)
        cumulative_concat_attention_weights = self.what_have_i_said(cumulative_concat_attention_weights) #(B, #chars, 128)

        # Broadcast sum the single mel timestep over all of our encoder timesteps 
        # (both attention weight features and encoder features)
        # And scale with tanh to get scores between -1 and 1, and project to a single value to comput energies

        # mel_proj (B, 1, 128) 
        # enc_proj_cache (B, #chars, 128)
        # cumulative_attention_weights (B, #chars, 128)
        energies = self.energy_proj(
            torch.tanh(
                mel_proj + self.enc_proj_cache + cumulative_concat_attention_weights
            )
        ).squeeze(-1) # final shape (B, #chars) for each time step mel input one number for one character
        
        # Mask out pad regions (dont want to weight pad tokens from encoder)
        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))
        
        return energies
    
    def forward(self, 
                mel_input, 
                encoder_output, 
                cumulative_concat_attention_weights, 
                mask=None):

        # Compute energies (B, #chars)
        energies = self._calculate_alignment_energies(mel_input, 
                                                      encoder_output, 
                                                      cumulative_concat_attention_weights, 
                                                      mask)
        
        # Convert to Probabilities (relation of our mel input to all the encoder outputs) #
        attention_weights = F.softmax(energies, dim=1)

        # Weighted average of our encoder states by the learned probabilities 

        # just like softmax(Q@K.T) for attention context
        # batch matrix multiplication torch.bmm
        # (B, 1, #chars) @ (B, #chars, embed_dim) to get the attention context vector 
        # (B, 1, #chars) @ (B, #chars, enc_embed_dim).T(1, 2) -> (B, 1, 512 = enc_embed_dim) -> squeeze the 2nd dim
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return attention_context, attention_weights
    

class PostNet(nn.Module):
    
    '''
    To take final generated Mel spec from LSTM and postprocess to allow for
    any missing details to be added in (learns the residual!) 
    '''
    def __init__(self, 
                 num_mels=80, 
                 postnet_num_convs = 5,
                 postnet_n_filters = 512,
                 postnet_kernel_size = 5,
                 postnet_dropout = 0.5):
        super(PostNet, self).__init__() 

        self.convs = nn.ModuleList()

        self.convs.append( # 1 time
            nn.Sequential(
                ConvNorm(in_channels=num_mels,
                         out_channels=postnet_n_filters,
                         kernel_size=postnet_kernel_size,
                         padding="same",
                         w_init_gain="tanh"),

                nn.BatchNorm1d(num_features=postnet_n_filters),
                nn.Tanh(),
                nn.Dropout(postnet_dropout)
            )
        )


        for _ in range(postnet_num_convs - 2): # 3 times
            
            self.convs.append(
                nn.Sequential(
                    ConvNorm(postnet_n_filters, 
                             postnet_n_filters, 
                             kernel_size=postnet_kernel_size, 
                             padding="same",
                             w_init_gain="tanh"), 

                    nn.BatchNorm1d(postnet_n_filters),
                    nn.Tanh(), 
                    nn.Dropout(postnet_dropout)
                )
            )

        # remining one time out of 5 times 
        self.convs.append(
            nn.Sequential(
                ConvNorm(postnet_n_filters,
                         num_mels,
                         kernel_size=postnet_kernel_size,
                         padding="same"),
                nn.BatchNorm1d(num_mels),
                nn.Dropout(postnet_dropout)
            )
        )


    def forward(self, x):

        #The postnet expects (batch x #chars x num_mels) 
        # but our convolution is almost the sequence dimension. 
        # So we transpose it first to (batch x num_mels x #chars) 
        # do our convolutions and then transpose back! 
        x = x.transpose(1, 2) 
        for conv_block in self.convs:
            x = conv_block(x)

        x = x.transpose(1, 2) #(B, #chars, num_mels)
        return x 
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        #new decoder step is coming from prenet 
        #current decoder step from prenet and attention context from encoder 
        # passing to lstm 
        # Predictions from previous timestep passed through a few linear layers
        self.prenet = Prenet(input_dim=config.num_mels,
                             prenet_dim = self.config.decoder_prenet_dim,
                             prenet_depth=self.config.decoder_prenet_depth)

        # LSTMs Module to Process Concatenated PreNet output and Attention Context Vector

        self.rnn = nn.ModuleList(
            [
                nn.LSTMCell(self.config.decoder_prenet_dim + self.config.encoder_embed_dim,
                            hidden_size=self.config.decoder_embed_dim),
                nn.LSTMCell(self.config.decoder_embed_dim + self.config.encoder_embed_dim, 
                            self.config.decoder_embed_dim)
            ]
        ) 

        # Local Sensitive Attention Module
        self.attention = LocalSensitiveAttention(attention_dim=self.config.attention_dim, 
                                                 decoder_hidden_size=self.config.decoder_embed_dim,
                                                 encoder_hidden_size=self.config.encoder_embed_dim, 
                                                 attention_n_filters=self.config.attention_location_n_filters, 
                                                 attention_kernel_size=self.config.attention_location_kernel_size)
    


        # predict next mel 
        # 2 linear layer after LSTM cell 
        # one for postnet another for stop token pred
        self.mel_proj =  LinearNorm(self.config.decoder_embed_dim + self.config.encoder_embed_dim, 
                                    self.config.num_mels)
        
        # whether its a stop token or not 
        # binary classification
        self.stop_proj = LinearNorm(self.config.decoder_embed_dim + self.config.encoder_emmbed_dim,
                                    1, w_init_gain="sigmoid")

        # Post Process Predicted Mel 
        self.postnet = PostNet(
            num_mels=config.num_mels, 
            postnet_num_convs=config.decoder_postnet_num_convs,
            postnet_n_filters=config.decoder_postnet_n_filters, 
            postnet_kernel_size=config.decoder_postnet_kernel_size,
            postnet_dropout=config.decoder_postnet_dropout_p
        )



    def _init_decoder(self, encoder_outputs, encoder_mask=None):

        # encoder outputs shape (B, #chars, embedding_dim)
        B, S, E = encoder_outputs.shape 
        device = encoder_outputs.device 

        # initialize memory for two lstm cells 
        # nn.LSTMCell(config.decoder_prenet_dim + config.encoder_embed_dim, config.decoder_embed_dim), 
        # nn.LSTMCell(config.decoder_embed_dim + config.encoder_embed_dim, config.decoder_embed_dim)

        self.h = [torch.zeros(B, self.config.decoder_embed_dim, device=device) for _ in range(2)] # for 2 lstm cells
        self.c = [torch.zeros(B, self.config.decoder_embed_dim, device=device) for _ in range(2)]

        # initialize cumulative attention 
        self.cumulative_attention_weights = torch.zeros(B, S, device=device) #(B, #chars) # only raw numbers 
        self.attn_weights = torch.zeros(B, S, device=device) #(B, #chars)
        self.attn_context = torch.zeros(B, self.config.encoder_embed_dim, device=device) #(B, 512) 

        # store encoder outputs 
        self.encoder_outputs = encoder_outputs #(B, #chars, 512) 
        self.encoder_mask = encoder_mask 


    def _bos_frame(self, B):
        #begining of sentence 
        # 0'th time step has no mel frame so only "0" 
        start_frame_zeros = torch.zeros(B, 1, self.config.num_mels) #(B, 1, 80) 
        return start_frame_zeros 
    

    def decode(self, mel_step):

        # mel_step is output from prenet layer 
        # mel_step (B, 1, 256) 256 config.decoder_prenet_dim 
        # everytime a new mel step 

        # nn.LSTMCell(config.decoder_prenet_dim + config.encoder_embed_dim, config.decoder_embed_dim),
        rnn_input_1 = torch.cat([mel_step, self.attn_context], dim=-1) #config.dec_prenet_dim + enc_embed_dim

        # pass the rnn input to 1st LSTMCell 
        self.h[0], self.c[0] = self.rnn[0](rnn_input_1, (self.h[0], self.c[0])) #pass hidden and memory to cell

        #dropout
        attn_hidden = F.dropout(self.h[0], self.config.attention_dropout_p, self.training) #(B, 1024)

        # concat cumulative_wattn_weights and prev_weights 
        # CAW + AW -> concat (B, 2, #chars)
        attn_weights_cat = torch.cat(
            [
                self.attn_weights.unsqueeze(1), #(B, 1, #chars)
                self.cumulative_attention_weights.unsqueeze(1) #(B, 1, #chars)
            ],
            dim = 1 

        )

        # attention context 

        new_attn_context, new_attn_weights = self.attention(
            attn_hidden, # this is the mel_input for energy calculation of shape (B, 1024)
            self.encoder_outputs, # (B, #chars, 512) 
            attn_weights_cat, # (B, 2, #chars) this will pass through LocationLayer 1D CNN 

        )

        self.attn_weights = new_attn_weights  #(B, #chars)
        self.cumulative_attention_weights += self.attn_weights  #(B, #chars)
        self.attn_context = new_attn_context # from the energy 

        # decoder input for 2nd lstm 
        # nn.LSTMCell(config.decoder_embed_dim + config.encoder_embed_dim, config.decoder_embed_dim)
        # attn_hidden = (B, 1024) + attn_context = (B, enc_embed_dim=512) 
        rnn_input_2 = torch.cat([attn_hidden, self.attn_context], dim=-1) 

        self.h[1], self.c[1] = self.rnn[1](rnn_input_2, (self.h[1], self.c[1]))
        decoder_hidden = F.dropout(self.h[1], self.config.decoder_dropout_p, self.training)

        # next mel pred 
        next_pred_input = torch.cat([
                decoder_hidden,
                self.attn_context
            ],dim=-1
        )

        mel_out = self.mel_proj(next_pred_input) # pass through linear layer for postnet and residual connection 
        stop_out = self.stop_proj(next_pred_input) # stop token 

        return mel_out, stop_out, new_attn_weights 
    

    def forward(self, 
                encoder_outputs,
                encoder_mask,
                mels,  # complete frames 
                decoder_mask):
        # When Decoding Start with Zero Feature Vector
        start_feature_vector = self._bos_frame(mels.shape[0]).to(encoder_outputs.device)  

        mels_w_start = torch.cat([start_feature_vector, mels], dim=1) # why?????
        self._init_decoder(encoder_outputs, encoder_mask)

        # Create lists to store Intermediate Outputs
        mel_outs, stop_tokens, attention_weights = [], [], []

        # Teacher forcing for T steps 
        T_Dec = mels.shape[1] 

        # project mel spec to prenet layer 
        mel_proj  = self.prenet(mels_w_start) #(B, T, 256) 

        for t in range(T_Dec):

            if t==0:
                self.attention.reset() # for each batch reset the enc_proj_cache

            step_input = mel_proj[:, t, :] #teach forcing real one not the predcited one  (B, 1, 256)
            mel_out, stop_out, attention_weight = self.decode(step_input) 
            mel_outs.append(mel_out)
            stop_tokens.append(stop_out)
            attention_weights.append(attention_weight)

        
        mel_outs = torch.stack(mel_outs, dim=1) #(B, T, 256)
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze() # (B, T, 1) -> ()
        attention_weights = torch.stack(attention_weights, dim=1)
        mel_residual = self.postnet(mel_outs)


        decoder_mask = decoder_mask.unsqueeze(-1).bool()
        mel_outs = mel_outs.masked_fill(decoder_mask, 0.0)
        mel_residual = mel_residual.masked_fill(decoder_mask, 0.0)
        attention_weights = attention_weights.masked_fill(decoder_mask, 0.0)
        stop_tokens = stop_tokens.masked_fill(decoder_mask.squeeze(), 1e3)


        return mel_outs, mel_residual, stop_tokens, attention_weights
    

    @torch.inference_mode()
    def inference(self, encoder_output, max_decode_steps=1000):

        start_feature_vector = self._bos_frame(B=1).squeeze(0) # starting with "0"
        self._init_decoder(encoder_output, encoder_mask=None)

        # Create lists to store Intermediate Outputs
        mel_outs, stop_outs, attention_weights = [], [], []

        _input = start_feature_vector 
        self.attention.reset() 

        while True:
            _input = self.prenet(_input) 
            mel_out, stop_out, attention_weight = self.decode(_input) 

            mel_outs.append(mel_out)
            stop_outs.append(stop_out)
            attention_weights.append(attention_weight)

            if torch.sigmoid(stop_out) > 0.5:
                break # stop token received 
            elif len(mel_outs) >= max_decode_steps:
                print("Reached Max Decoder Steps")
                break 

            _input = mel_out 

        mel_outs = torch.stack(mel_outs, dim=1)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)
        mel_residual = self.postnet(mel_outs)

        return mel_outs, mel_residual, stop_outs, attention_weights
    

class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, text, input_lengths, mels, encoder_mask, decoder_mask):

        encoder_padded_outputs = self.encoder(text, input_lengths)
        mel_outs, mel_residual, stop_tokens, attention_weights = self.decoder(
            encoder_padded_outputs, encoder_mask, mels, decoder_mask
        )

        mel_postnet_out = mel_outs + mel_residual

        return mel_outs, mel_postnet_out, stop_tokens, attention_weights 
    
    @torch.inference_mode()
    def inference(self, text, max_decode_steps=1000):
        
        if text.ndim == 1:
            text = text.unsqueeze(0)

        assert text.shape[0] == 1, "Inference only written for Batch Size of 1"
        encoder_outputs = self.encoder(text)
        mel_outs, mel_residual, stop_outs, attention_weights = self.decoder.inference(
            encoder_outputs, max_decode_steps=max_decode_steps
        )

        mel_postnet_out = mel_outs + mel_residual

        return mel_postnet_out, attention_weights   
    

if __name__ == "__main__":

    from dataset import TTSDataset, TTSColator

    dataset = TTSDataset("/home/cmi_10101/Documents/coding/" \
    "pytorch/architecture-implementation/tts/" \
    "LJSpeech-1.1/train_metadata.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=TTSColator())
    for text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask in loader:

        config = Tacotron2Config()
        model = Tacotron2(config)
        print(model)
        # decoder(encoded_outputs, )

        break   






    








    
        
        

