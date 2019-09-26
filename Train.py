import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from LSR_GAN import VAE,Discriminator
from tqdm import tqdm


torch.manual_seed(13)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='')
parser.add_argument('-batch_size', type=int, default=32, help='batch size for trainset and testset (default:32)')
parser.add_argument('-vae_epochs', type=int, default=100, help='epochs to train for vae (default:100)')
parser.add_argument('-gan_epochs', type=int, default=500, help='epochs to train for lsr-gan (default:500)')
parser.add_argument('-lr', type=float, default=0.0002, help='learning rate (default: 0.0002)')
parser.add_argument('-beta', type=int, default=1, help='beta value for vae (default: 1)')
parser.add_argument('-lam', type=int, default=1, help='lambda value for lsr-gan (default: 1)')
args = parser.parse_args()

trainset = CIFAR10(".", train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
testset = CIFAR10(".", train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

                           

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def vae_loss(recon_x, x, mu, logvar):
    
    BCE = (torch.log((recon_x-x).pow(2).sum()))*0.5*3*32*32*x.shape[0]
    KLD = -0.5 * torch.sum(1 + logvar-mu.pow(2) - logvar.exp())*args.beta

    return BCE+KLD,BCE,KLD


def log_loss(z,mu,logvar):
    pdf = (((z - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5*logvar + math.log(math.sqrt(2 * math.pi))).mean()*args.lam
    return pdf

def freeze(module: nn.Module):
    for p in model_1.encoder.parameters():
        p.requires_grad = False


model_1 = VAE().to(device)
opt_VAE = optim.Adam(model_1.parameters(),lr = args.lr, betas=(0.5, 0.999))
writer = SummaryWriter('runs/VAE')


for epoch in tqdm(range(args.vae_epochs)):
    model_1.train()
    train_loss = 0
    train_kl = 0
    train_bce = 0


    for i, (data,_) in enumerate(trainloader):
        data = data.to(device)
        opt_VAE.zero_grad()
        recon, mu, logvar= model_1(data)
        loss,bce,kl = vae_loss(recon, data, mu, logvar)
        loss.backward()
        opt_VAE.step()

        train_loss += loss.item()
        train_bce += bce.item()
        train_kl += kl.item()
        
        
    train_loss = train_loss / len(trainloader.dataset)
    train_bce = train_bce / len(trainloader.dataset)
    train_kl = train_kl / len(trainloader.dataset)
    writer.add_scalar('KL', train_kl, epoch)
    writer.add_scalar('BCE', train_bce, epoch)
    writer.add_scalar('loss', train_loss, epoch)

    
    model_1.eval()
    for i, (data,_) in enumerate(testloader):
        data = data.to(device)
        recon, mu, logvar= model_1(data)
        if i == 0:
            image = utils.make_grid(
                torch.cat([recon,data]),
                nrow=16,
                normalize=True
            )
            writer.add_image('recon', image, epoch)          
            break
            
    with torch.no_grad():        
        sample = torch.randn(128,128).to(device)
        sample = model_1.decoder(sample)

        image_sam = utils.make_grid(
            sample,
            nrow=16,
            normalize=True
        )
        writer.add_image('sample',image_sam,epoch)


torch.save(model_1.state_dict(), 'VAE')
freeze(model_1.encoder)
model_1.encoder.eval()


writer = SummaryWriter('runs/GAN')


discriminator = Discriminator().to(device)
optimizer_D = optim.Adam(discriminator.parameters(),lr = args.lr, betas=(0.5, 0.999))
optimizer_G = optim.Adam(model_1.decoder.parameters(),lr = args.lr, betas=(0.5, 0.999))
discriminator.apply(weights_init)


for epoch in tqdm(range(args.gan_epochs)):
    model_1.decoder.train()
    Gloss = 0
    Dloss = 0
    logpro = 0
    for batch_idx, (data,_) in enumerate(trainloader):
        valid = torch.ones(data.shape[0],3,2,2).to(device)
        fake = torch.zeros(data.shape[0],3,2,2).to(device)
        sample = torch.randn(data.shape[0],128).to(device)
        
        G_fake = model_1.decoder(sample)
        mu, logvar = model_1.encoder(G_fake)
        data = data.to(device)
        
        optimizer_D.zero_grad()
        D_fake = discriminator(G_fake.detach())
        D_true = discriminator(data)
        
        D_loss_fake = F.binary_cross_entropy(D_fake, fake)
        D_loss_true = F.binary_cross_entropy(D_true, valid)
        D_loss = D_loss_fake + D_loss_true
        D_loss.backward()
        Dloss += D_loss.item()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        D_fake = discriminator(G_fake)
        log_pro = log_loss(sample,mu,logvar)
        G_loss = F.binary_cross_entropy(D_fake, valid)
        (G_loss+log_pro).backward()
        Gloss += G_loss.item()
        logpro += log_pro.item()
        optimizer_G.step()
        
        
    writer.add_scalar('Dloss',Dloss/len(trainloader.dataset), epoch)
    writer.add_scalar('Gloss',Gloss/len(trainloader.dataset), epoch)
    writer.add_scalar('logpro',logpro/len(trainloader.dataset), epoch)
    
    
    model_1.eval()
    for i, (data,_) in enumerate(testloader):
        data = data.to(device)
        mu, logvar= model_1.encoder(data)
        recon = model_1.decoder(mu)
        if i == 0:
            image = utils.make_grid(
                    torch.cat([recon,data]),
                    nrow=16,
                    normalize=True
                )
            writer.add_image('recon', image, epoch)
        break
        
    with torch.no_grad():       
        sample = torch.randn(128,128).to(device)
        output = model_1.decoder(sample)
        image_sam = utils.make_grid(
                output,
                nrow=16,
                normalize=True)
        writer.add_image('sample',image_sam,epoch)


torch.save(discriminator.state_dict(), 'Discriminator')
torch.save(model_1.decoder.state_dict(), 'Generator')




