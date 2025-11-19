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

### Import Model ###
from vae import LinearVAE 

### Prep Dataset ###
tensor_transforms = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
)
train_set = MNIST(r"C:\\Users\ASUS\\Documents\\coding\\Model-implementation\\encoder-decoder-network\\data\\mnist", 
                train=True, 
                transform=tensor_transforms,
                download=True)
test_set = MNIST(r"C:\\Users\ASUS\\Documents\\coding\\Model-implementation\\encoder-decoder-network\\data\\mnist", 
                train=False, 
                transform=tensor_transforms,
                download=True)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0) 

model = LinearVAE(latent_dim=2).to(device)
### Optimizer ###
optimizer = optim.Adam(model.parameters(), lr=1e-3) 
### Loss Function ###
def vae_loss(x, x_recon, mean, log_var, kl_weight=1, recon_weight=1):

    # compute mse for every pixel [B, C, H, W]
    pixel_mse = (x - x_recon)**2 
    pixel_mse = pixel_mse.view(pixel_mse.shape[0], -1).mean(dim=1) # mean over all but batch dimension 
    # sum up pixel loss per image average across batch4
    recon_loss = pixel_mse.sum(axis=-1).mean()

        ### Compute KL Per Image and Sum Across Flattened Latent ###
    kl = (1 + log_var - mean**2 - torch.exp(log_var)).flatten(1)
    kl_per_image = - 0.5 * torch.sum(kl, dim=-1)

    ### Average KL Across the Batch ###
    kl_loss = torch.mean(kl_per_image)
    
    return recon_weight*recon_loss + kl_weight*kl_loss 

def train(model,
          kl_weight,
          train_set,
          test_set,
          batch_size, 
          training_iterations, 
          evaluation_iterations,
          model_type="VAE"):

    if model_type != "VAE": kl_weight = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_loss = []
    evaluation_loss = []
    
    encoded_data_per_eval = []
    train_losses = []
    evaluation_losses = []

    pbar = tqdm(range(training_iterations))
    
    train = True
    
    step_counter = 0
    while train:

        for images, labels in trainloader:
            
            images = images.to(device)

            # if model_type == "VAE":
            encoded, decoded, mu, logvar = model(images)
            loss = vae_loss(images, decoded, mu, logvar, kl_weight)
                
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step_counter % evaluation_iterations == 0:
                
                model.eval()
                
                encoded_evaluations = []

                for images, labels in testloader:

                    images = images.to(device)
                    # if model_type == "VAE":
                    encoded, decoded, mu, logvar = model(images)
                    loss = vae_loss(images, decoded, mu, logvar, kl_weight)

                    evaluation_loss.append(loss.item())

                    encoded, labels = encoded.cpu().flatten(1), labels.reshape(-1,1)
                    
                    encoded_evaluations.append(torch.cat((encoded, labels), axis=-1))


                encoded_data_per_eval.append(torch.concatenate(encoded_evaluations).detach())

                train_loss = np.mean(train_loss)
                evaluation_loss = np.mean(evaluation_loss)

                train_losses.append(train_loss)
                evaluation_losses.append(evaluation_loss)
                
                train_loss = []
                evaluation_loss = []

                model.train()

            step_counter += 1
            pbar.update(1)
           
            
            if step_counter >= training_iterations:
                print("Completed Training!")
                train = False
                break

    encoded_data_per_eval = [np.array(i) for i in encoded_data_per_eval]

    print("Final Training Loss", train_losses[-1])
    print("Final Evaluation Loss", evaluation_losses[-1])
    
    return model, train_losses, evaluation_losses, encoded_data_per_eval

def main():
    models = []
    encoded_datas = []

    # Train our Variational AutoEncoders with different KL Weights
    kl_weights = [1, 100]
    for kl_weight in kl_weights:
        print(f"KL Weight:", kl_weight)
        model = LinearVAE()
        model, train_losses, evaluation_losses, encoded_data_per_eval = train(
            model,
            kl_weight=kl_weight,
            train_set=train_set,
            test_set=test_set,
            batch_size=64,
            training_iterations=25000,
            evaluation_iterations=250,
            model_type="VAE"
        )

        models.append(model)
        encoded_datas.append(encoded_data_per_eval)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()   # good practice on Windows
    main()
