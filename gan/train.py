import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# Models import 
from model import MNISTDiscriminator, MNISTGenerator, MNISTConditionalDiscriminator, MNISTConditionalGenerator


### Stuff to Visualize the Latent Space ###
from celluloid import Camera
from IPython.display import HTML

### Seed Everything ###
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

### Set Device ###
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"Using device: {device}") 

latent_dimension = 100
batch_size = 64
epochs = 200
generator_lr = 3e-4 
discriminator_lr = 3e-4 
CONDITIONAL = True

# Define models
if CONDITIONAL: 
    generator = MNISTConditionalGenerator().to(device)
    discriminator = MNISTConditionalDiscriminator().to(device)
else:
    generator = MNISTGenerator(latent_dim=latent_dimension).to(device=device)
    discriminator = MNISTDiscriminator().to(device=device)

# Define optimizers 

gen_optimizer = optim.AdamW(generator.parameters(), lr=generator_lr)
disc_optimizer = optim.AdamW(discriminator.parameters(), lr=discriminator_lr)

# Define Datasets 
tensor2image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = MNIST(r"C:\\Users\ASUS\\Documents\\coding\\Model-implementation\\encoder-decoder-network\\data\\mnist", 
                train=True, 
                transform=tensor2image_transforms,
                download=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

def train_unconditional_gan(gen, disc, 
                            gen_optim, disc_optim, 
                            dataloader, label_smoothing = 0.05,
                            epochs = 200,
                            device = "cpu",
                            plot_generation_freq=50,
                            plot_loss_freq=20,
                            num_gens=10):
    # Define the loss function 
    ''' 
    This loss combines a Sigmoid layer and the BCELoss in one single class. 
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
    by combining the operations into one layer, 
    we take advantage of the log-sum-exp trick for numerical stability.
    '''
    loss_func = nn.BCEWithLogitsLoss()
    gen_losses, disc_losses = [], [] 

    for epoch in tqdm(range(epochs)):
        gen_epoch_losses = []
        disc_epoch_losses = []

        for images, _ in dataloader:
            B = images.shape[0] 
            # images are real images 
            images = images.to(device)

            # Train Discriminator   
            noise = torch.randn(B, latent_dimension, device=device)

            #  Create Labels for Discriminator with label smoothing
            # we are not using MNIST data labels we are just assigning true as 1 and fake as 0 that's it!!
            generated_labels = torch.zeros(B, 1, device=device) + label_smoothing
            true_labels = torch.ones(B, 1, device=device) - label_smoothing 

            # We are training Discriminator so remove G(z) from comp graph 
            generated_images = gen(noise).detach() 

            # pass real and generated images to Disc 
            real_disc_pred = disc(images)
            gen_disc_pred = disc(generated_images) 

            # compute Disc loss 
            real_loss = loss_func(real_disc_pred, true_labels)
            fake_loss = loss_func(gen_disc_pred, generated_labels) 
            disc_loss = (real_loss+fake_loss) / 2 
            disc_epoch_losses.append(disc_loss.item()) 

            # update Discriminator 
            disc_optim.zero_grad() # before training step of each model zeroing grads as its accumuluated of previous 
            disc_loss.backward() 
            disc_optim.update() 


            # Train Generator 
            # sample noise for generation
            noise = torch.randn(batch_size, latent_dimension, device=device) 
            generated_images = gen(noise) 
            gen_disc_pred = disc(generated_images)   # pass to Disc to Fool!!
            # compute gen loss 
            gen_loss = loss_func(gen_disc_pred, true_labels) # pretend as true images 
            gen_epoch_losses.append(gen_loss.item())

            gen_optim.zero_grad() # before training step of each model zeroing grads as its loss accumuluated of previously 
            gen_loss.backward() # this loss contain both gen and disc only the naming is gen_loss dont get confuse
            gen_optim.step()

        gen_epoch_losses = np.mean(gen_epoch_losses)
        disc_epoch_losses = np.mean(disc_epoch_losses) 

        if epoch % plot_loss_freq == 0:
            print(f"Epoch: {epoch}/{epochs} | Generator Loss: {gen_epoch_losses} | Discriminator Loss: {disc_epoch_losses}")
    
        gen_losses.append(gen_epoch_losses)
        disc_losses.append(disc_epoch_losses)
    
        if epoch % plot_generation_freq == 0:
            generator.eval()
            with torch.no_grad():
                noise_sample = torch.randn(num_gens, latent_dimension, device=device)
                generated_imgs = generator(noise_sample).to("cpu")
        
                fig, ax = plt.subplots(1,num_gens, figsize=(15,5))
        
                for i in range(num_gens):
                    img = (generated_imgs[i].squeeze() + 1)/2
                    ax[i].imshow(img.numpy(), cmap="gray")
                    ax[i].set_axis_off()
        
                plt.show()
                
            generator.train()

    return gen, disc, gen_losses, disc_losses


def train_conditional_gan(
                        generator, 
                        discriminator, 
                        generator_optimizer, 
                        discriminator_optimizer, 
                        dataloader,
                        label_smoothing=0.05,
                        epochs=200,
                        device="cpu", 
                        plot_generation_freq=50,
                        plot_loss_freq=20,
                        num_classes=10
                        ):
    loss_func  = nn.BCEWithLogitsLoss()
    gen_losses, disc_losses = [], [] 

    for epoch in tqdm(range(epochs)):
        generator_epoch_losses = []
        discriminator_epoch_losses = []

        for images, true_digits in dataloader:
            B = images.shape[0] 
            images = images.to(device) # real images 
            true_digits = true_digits.to(device) # true labels of MNIST digit  

            # Train Discriminator 
            # NEW: Sample noise/random digits for Generation 
            rand_digits = torch.randint(0, num_classes, size=(B, ), device=device)
            noise = torch.randn(B, latent_dimension, device=device) 

            # create labels 
            generated_labels = torch.zeros(B, 1, device=device) + label_smoothing 
            true_labels = torch.ones(B, 1, device=device) - label_smoothing 

            # sample G(z) and remove it from comp graph 
            generated_images = generator(noise, rand_digits).detach() 

            # pass to Discriminator 
            real_discriminator_pred = discriminator(images, true_digits)
            gen_discriminator_pred = discriminator(generated_images, rand_digits) 

            # compute Disc loss 
            real_loss = loss_func(real_discriminator_pred, true_labels) 
            fake_loss = loss_func(gen_discriminator_pred, generated_labels) 
            discriminator_loss = (real_loss + fake_loss) / 2 
            discriminator_epoch_losses.append(discriminator_loss.item())

            # update the weights with optimizer 
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step() 

            # Train Generator 
            # Sample noise for Generation 
            rand_digits = torch.randint(0,num_classes, size=(B,), device=device)
            noise = torch.randn(B, latent_dimension, device=device)
            generated_images = generator(noise, rand_digits) 
            gen_discriminator_pred = discriminator(generated_images, rand_digits)

            # loss 
            generator_loss = loss_func(gen_discriminator_pred, true_labels) # pretend as true so true labels 
            generator_epoch_losses.append(generator_loss.item()) 

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        generator_epoch_losses = np.mean(generator_epoch_losses)
        discriminator_epoch_losses = np.mean(discriminator_epoch_losses)

        if epoch % plot_loss_freq == 0:
            print(f"Epoch: {epoch}/{epochs} | Generator Loss: {generator_epoch_losses} | Discriminator Loss: {discriminator_epoch_losses}")
    
        gen_losses.append(generator_epoch_losses)
        disc_losses.append(discriminator_epoch_losses)
    
        if epoch % plot_generation_freq == 0:
            generator.eval()
            with torch.no_grad():
                digits = torch.arange(num_classes, device=device)
                noise_sample = torch.randn(num_classes, latent_dimension, device=device)
                generated_imgs = generator(noise_sample, digits).to("cpu")
        
                fig, ax = plt.subplots(1,num_classes, figsize=(15,5))
        
                for i in range(num_classes):
                    img = (generated_imgs[i].squeeze() + 1)/2
                    ax[i].imshow(img.numpy(), cmap="gray")
                    ax[i].set_axis_off()
        
                plt.show()
                
            generator.train()

    return generator, discriminator, gen_losses, disc_losses 


# generator, discriminator, gen_losses, disc_losses = train_conditional_gan(generator=generator, 
#                                                                           discriminator=discriminator, 
#                                                                           generator_optimizer=gen_optimizer, 
#                                                                           discriminator_optimizer=disc_optimizer, 
#                                                                           dataloader=trainloader,
#                                                                           epochs=200,
#                                                                           device="cuda", 
#                                                                           plot_generation_freq=40,
#                                                                           plot_loss_freq=40)