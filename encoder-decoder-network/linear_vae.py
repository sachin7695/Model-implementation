import torch 
import torch.nn as nn
import torch.nn.functional as F 

class LinearVAE(nn.Module):

    def __init__(self, latent_dim=2):
        super(LinearVAE, self).__init__() 

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.fn_mu = nn.Linear(32, latent_dim)
        self.log_var = nn.Linear(32, latent_dim) 

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )


    def forward_enc(self, x):
        x = self.encoder(x)

        # compute mu and sigma 
        mu = self.fn_mu(x) 
        log_var = self.log_var(x) 

        # sample with reparameterization trick 
        sigma = torch.exp(0.5 * log_var) 
        noise = torch.randn_like(sigma, device=sigma.device) 
        z = mu + sigma * noise # mu + sigma * epsilon episilon ~ N(0, 1) 
        return z, mu, log_var 
    
    def forward_dec(self, z): 
        return self.decoder(z) 
    
    def forward(self, x):
        b, c, h, w = x.shape 
        x = x.flatten(1) # 1d vector flattened 

        #pass through encoder 
        z, mu, log_var = self.forward_enc(x) 
        #pass through decoder
        x_recon = self.forward_dec(z) 
        # put back to original shape 
        x_recon = x_recon.reshape(b, c, h, w) 
        return z, x_recon, mu, log_var,  
    

if __name__ == "__main__":
    model = LinearVAE(latent_dim=2) 
    x = torch.randn(4, 1, 32, 32) 
    z, x_recon,mu, log_var  = model(x) 
    print("Input shape: ", x.shape) 
    print("Latent shape: ", z.shape) 
    print("Reconstructed shape: ", x_recon.shape)
