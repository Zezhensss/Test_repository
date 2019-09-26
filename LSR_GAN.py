import torch
from torch import nn, optim
from torch.nn import functional as F
from Resblock import ResnetBlock, ResnetBlockTranspose

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.main = nn.Sequential(
            ResnetBlock(3, 64),
            # state size. (64) x 32 x 32
            ResnetBlock(64, 64*2),
            # state size. (64*2) x 16 x 16
            ResnetBlock(64*2, 32),
            # state size. (32) x 8 x 8
        )
        self.fc = nn.Sequential(nn.Linear(512, 128, bias=False),
                                nn.BatchNorm1d(128),
                                nn.ReLU(True))
        self.mu = nn.Linear(128, 128)
        self.logvar = nn.Linear(128, 128)
    
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.fc = nn.Sequential(nn.Linear(128, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(True))
        self.main = nn.Sequential(
            ResnetBlockTranspose(32, 64*2),
            # state size. (ngf*4) x 16 x 16
            ResnetBlockTranspose(64*2, 64),
            # state size. (ngf*2) x 32 x 32
            ResnetBlockTranspose(64, 32),
            # state size. (ngf) x 64 x 64
            nn.Conv2d( 32, 3, (1,1), bias=False),
            nn.Tanh()

        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0],32,4,4)
        return self.main(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,x):
        mu,logvar = self.encoder(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        recon = self.decoder(z)
        
        return recon, mu, logvar


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 4, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.main(input)