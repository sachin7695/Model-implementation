import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
from dataclasses import dataclass 


@dataclass
class Tacotron2Config:

    ### Mel Input Features ###
    num_mels: int = 80 

    ### Character Embeddings ###
    character_embed_dim: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    ### Encoder config ###
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5
    
    ### Decoder Config ###
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout_p: float = 0.5
    decoder_dropout_p: float = 0.1

    ### Attention Config ###
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

            #(80, 256) -> (256, 256) -> output dim 256 dim hidden vector of acoustic feature 

    def forward(self, x): 
        for layer in self.layers:
            ### Even during inference we leave this dropout enabled to "introduce output variation" ###
            x = F.dropout(layer(x), p=self.dropout_p, training=True)
        return x #B, T , 256 
    
class LocationLayer(nn.Module):

    def __init__(self,
                 attention_n_filters,
                 attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()

        self.conv = ConvNorm(
            in_channels=2 , #attention weights shape B, 2, attention_dim 
            out_channels=attention_n_filters, kernel_size=attention_kernel_size,
            padding="same",
            bias=False
        )

        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights):
        #Attention_weights.shape Bx2xattention_dim 
        attention_weights = self.conv(attention_weights).permute(0, 2, 1).contiguous()
        #attention_weights.shape Bxattention_Dimx32
        attention_weights = self.proj(attention_weights) #Bxattention_dimxattention_dim
        return attention_weights

class LocalSensitiveAttention(nn.Module):

    def __init__(self, 
                 attention_dim,
                 decoder_hidden_size,
                 encoder_hidden_size,
                 attention_n_filters,
                 attention_kernel_size):
        super(LocalSensitiveAttention, self).__init__()

        self.in_proj = LinearNorm(decoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")
        self.enc_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")


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
                                    mel_input, 
                                    encoder_output, 
                                    cumulative_attention_weights, 
                                    mask=None):

        ### Take our previous step of the mel sequence and project it (B x 1 x attention_dim)
        mel_proj = self.in_proj(mel_input).unsqueeze(1)

        ### Take our entire encoder output and project it (B x encoder_len x attention_dim)
        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)

        ### Look at our attention weight history to understand where the model has already placed attention 
        cumulative_attention_weights = self.what_have_i_said(cumulative_attention_weights)

        ### Broadcast sum the single mel timestep over all of our encoder timesteps (both attention weight features and encoder features)
        ### And scale with tanh to get scores between -1 and 1, and project to a single value to comput energies
        energies = self.energy_proj(
            torch.tanh(
                mel_proj + self.enc_proj_cache + cumulative_attention_weights
            )
        ).squeeze(-1)
        
        ### Mask out pad regions (dont want to weight pad tokens from encoder)
        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))
        
        return energies
    
    def forward(self, 
                mel_input, 
                encoder_output, 
                cumulative_attention_weights, 
                mask=None):

        ### Compute energies ###
        energies = self._calculate_alignment_energies(mel_input, 
                                                      encoder_output, 
                                                      cumulative_attention_weights, 
                                                      mask)
        
        ### Convert to Probabilities (relation of our mel input to all the encoder outputs) ###
        attention_weights = F.softmax(energies, dim=1)

        ### Weighted average of our encoder states by the learned probabilities 
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return attention_context, attention_weights


    
        
        

