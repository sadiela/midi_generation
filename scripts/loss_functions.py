import torch 
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(x_hat, x, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + kld

def bce_recon_loss(x_hat, x):
    return nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')