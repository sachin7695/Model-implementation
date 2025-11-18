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
    

class ConvolutionalVAE(nn.Module):

    def __init__(self, in_channels, channels_bottleneck=4):
        super(ConvolutionalVAE, self).__init__()

        self.bottleneck = channels_bottleneck 
        self.in_channels = in_channels

        self.encoder_conv = nn.Sequential(

            # Encoding 
            nn.Conv2d(in_channels=in_channels,out_channels=8,kernel_size=3,
                      stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8), # out channels (N, C, H, W) prev
            nn.ReLU(), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=self.bottleneck,kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.bottlneck),
            nn.ReLU() 
        )

        self.conv_mu = nn.Conv2d(in_channels=self.bottleneck, out_channels=self.bottleneck,
                                 kernel_size=3, stride=1,padding="same")
        self.conv_logvar = nn.Conv2d(in_channels=self.bottleneck,out_channels=self.bottleneck,
                                    kernel_size=3, stride=1,padding="same")
        

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.bottleneck,out_channels=16,
                               kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward_enc(self, x):
        B, C, H, W = x.shape 
        conv_enc = self.encoder_conv(x) 

        mu = self.conv_mu(conv_enc) 
        logvar = self.conv_logvar(conv_enc) 

        # sample with reparametrization trick 
        sigma = torch.exp(0.5*logvar)
        noise = torch.randn_like(sigma, device=sigma.device)
        z = mu + sigma*noise 

        return z, mu, logvar
     
    def forward_dec(self, x):
        B = x.shape[0] 
        x = x.reshape((B, self.bottleneck, 4, 4))
        conv_dec = self.forward_dec(x)
        return conv_dec 
    
    def forward(self, x):
        B, C, H, W = x.shape 
        z, mu, logvar = self.forward_enc(x) 
        dec = self.forward_dec(z) 
        return z, dec, mu, logvar 
    


if __name__ == "__main__":
    # model = LinearVAE(latent_dim=2)
    model = ConvolutionalVAE()

    x = torch.randn(4, 1, 32, 32) 
    z, x_recon,mu, log_var  = model(x) 
    print("Input shape: ", x.shape) 
    print("Latent shape: ", z.shape) 
    print("Reconstructed shape: ", x_recon.shape)
