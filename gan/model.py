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
    

class MNISTConditionalGenerator(nn.Module):
    def __init__(self, latent_dim = 100, num_embeddings = 10, embedding_dim = 16):
        super(MNISTConditionalGenerator, self).__init__()
        ''' 
        Generator embedding:
        It learns how a label should look in image space.
        The embedding guides the generator to produce visual features of that digit.
        '''
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim=embedding_dim) # (10, 16)

        self.generator = nn.Sequential(
            nn.Linear(latent_dim+embedding_dim, 256), # conctanating noise and label embedding 
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        B, latent_dim = noise.shape 
        # digit embeddings
        embeddings = self.embeddings(labels) 
        noise = torch.cat([noise, embeddings], dim=-1) 
        generated = self.generator(noise)
        generated = generated.view(B, 1, 28, 28) 
        return generated 
    
class MNISTConditionalDiscriminator(nn.Module):
    def __init__(self, num_embeddings = 10, embedding_dim=16):
        '''
        Dsicriminator embedding:
        It learns what the label means relative to real images.
        It uses the embedding to check
        NOTE: different kinds of information. Using a single shared embedding would actually hurt training.
        '''

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(784+embedding_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x, labels):
        B, C, H, W = x.shape 
        embeddings = self.embeddings(labels) 
        x = x.reshape(B, -1) # flattening the images 
        x = torch.cat([x, embeddings], dim=-1)
        return self.discriminator(x) 

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=False):
        super().__init__()
        if interpolate:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=3, 
                          padding="same")
            )

        else:

            # I just messed with the padding and output padding to ensure that
            # (7x7) goes to (14x14) and (14x14) goes to (28x28)
            # So we always have a 2x upsample
            self.upsample = nn.ConvTranspose2d(in_channels, 
                                               out_channels,
                                               kernel_size=3, 
                                               stride=2, 
                                               padding=1, 
                                               output_padding=1)
            

    def forward(self, x):
        return self.upsample(x)

class ConvMNISTConditionalGenerator(nn.Module):
    def __init__(self, 
                 in_channels = 128,
                 start_dim = 7,
                 latent_dimension=100,
                 num_embeddings = 10,
                 embedding_dim = 16,
                 interpolate = False):
        super().__init__() 
        self.start_dim  = start_dim 
        self.in_channels = in_channels 

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim) 
        self.lin2img = nn.Linear(latent_dimension+embedding_dim, in_channels * start_dim * start_dim) 

        self.generator = nn.Sequential(
            UpsampleBlock(in_channels, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,kernel_size=3, padding="same"), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            UpsampleBlock(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,1,kernel_size=3, padding="same"), 
            nn.Tanh()
        )

    def forward(self, noise, labels):
        B = noise.shape[0] 
        embeddings = self.embeddings(labels)
        noise = torch.cat([noise, embeddings], dim=-1)
        # Project Noise to Img Space
        noise = self.lin2img(noise) 
        noise = noise.reshape(B, self.in_channels, self.start_dim, self.start_dim) # Reshape Noise to Image Shape
        generated = self.generator(noise) 
        return generated.reshape(B, 1, 28, 28)

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


        
