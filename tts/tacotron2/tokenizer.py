import torch 

class Tokenizer:

    def __init__(self):

        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.chars = [self.pad_token, self.eos_token, self.unk_token] + \
            list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ')
        
        self.char_to_idx = {ch:i for i, ch in enumerate(self.chars)}
        self.eos_token_idx = self.char_to_idx[self.eos_token]
        self.pad_token_idx = self.char_to_idx[self.pad_token]
        self.unk_token_idx = self.char_to_idx[self.unk_token]

        self.vocab_size = len(self.chars)


    def encode(self, text, return_tensor=True):
        tokens  = []
        for ch in text:
            tokens.append(self.char_to_idx.get(ch, self.unk_token_idx))
        tokens.append(self.eos_token_idx)

        if return_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long) 


        return tokens 
