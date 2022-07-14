import torch 
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(x_hat, x, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + kld

def kld(mean, log_var):
    return - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

def bce_recon_loss(x_hat, x):
    return nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')

def calculate_recon_error(x_hat, x, lossfunc='mse', lam=1):
    if lossfunc=='mse':
        recon_error = F.mse_loss(x_hat, x)
    elif lossfunc=='l1reg':
        recon_error = F.mse_loss(x_hat, x) + (1.0/x.shape[0])*lam*torch.norm(x_hat, p=1) # +  ADD L1 norm
    elif lossfunc=='bce':
        recon_error=bce_recon_loss(x_hat, x)
    else: # loss function = mae
        recon_error = F.l1_loss(x_hat, x)

    return recon_error