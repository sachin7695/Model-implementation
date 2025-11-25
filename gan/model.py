import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.datasets import MNIST 
from torchvision.transforms import transforms 
from torch.utils.data import  DataLoader  
import torch.optim as optim 

import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 

# Simple Linear unconditional MNIST GAN 

# Generator 
class MNISTGenerator(nn.Module):

    ''' 
    The generator will take in some vector of noise (the dimension of which is the latent dimension) 
    and then use a few linear layers to project to a vector of dimension 784 
    (as that is how many pixels we have in our MNIST Images). We end the model with the TanH function, 
    as it will scale the pixels between -1 and 1, which is also what our MNIST images will be scaled to
    '''


    def __init__(self, latent_dim=784):
        super(MNISTGenerator, self).__init__() 
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784), #784 pixels as 28x28 in our MNIST datasets
            nn.Tanh() # scaling between [-1, 1] thats how our MNIST is also scaled 
        )

    def forward(self, noise):
        B, latent_dim = noise.shape # dim is 256
        # B = noise.shape[0] 
        generated = self.generator(noise)
        generated = generated.view(B, 1, 28, 28) # reshaping in the real sample images 28x28 
        return generated 
        
class MNISTDiscriminator(nn.Module):

    ''' 
    The discriminator will take in images of shape (Batch x 1 x 28 x 28) which are our MNIST dimensions.
    It will then flatten the images and use a stack of linear layers to predict 1 output. 
    This will then go to our BCELoss to predict if the image is real or fake
    '''
    def __init__(self):
        super(MNISTDiscriminator, self).__init__() 

        self.discriminator = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1) # Real or fake 
        )

    def forward(self, x):
        B, C, H, W = x.shape  # gray scale image 
        x = x.view(B, C*H*W) # (B, 784) 
        return self.discriminator(x)
    
if __name__ == "__main__":
    model_1 = MNISTDiscriminator()
    model_2 = MNISTGenerator()
    x_1 = torch.randn(size=(4, 1, 28, 28))
    x_2 = torch.randn(size=(4, 784))
    res_1 = model_1(x_1) 
    res_2 = model_2(x_2)
    print(res_1)
    print(f"shape after Discriminator: {res_1.shape}")
    print(res_2)  
    print(f"shape after Generator: {res_2.shape}")


        
