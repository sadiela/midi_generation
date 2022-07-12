'''
This file contains the VQ-VAE model class
'''
###########
# Imports #
###########
#import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick device


# is reconstruction error going down? 
# Run w/ more data
# Increase # of embedding vectors? 


class VAE_Encoder(nn.Module):
  def __init__(self, in_channels, hidden_dim, latent_dim):
    super(VAE_Encoder, self).__init__()
    self._conv_1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=8,
                              kernel_size=(1,32))
    self._conv_2 = nn.Conv2d(in_channels=8,
                              out_channels=4,
                              kernel_size=(1,64))
    self._conv_3 = nn.Conv2d(in_channels=4,
                              out_channels=1,
                              kernel_size=(1,8))
    self.pool = nn.MaxPool2d((1, 2))
    self.FC_mean = nn.Linear(hidden_dim, latent_dim)
    self.FC_var = nn.Linear(hidden_dim, latent_dim)

  def forward(self, inputs):
    logging.debug("FORWARD")
    x = self._conv_1(inputs)
    x = F.relu(x)
    logging.debug("After first conv", x.shape)

    x = self._conv_2(x)
    #x = self.pool(x)
    x = F.relu(x)

    logging.debug("After second conv", x.shape)

    x = self._conv_3(x)

    logging.debug("After third conv", x.shape)

    batch_size = x.shape[0]
    x_dim = x.shape[2]*x.shape[3]

    x = x.view(batch_size, x_dim)
    logging.debug(x.shape)

    mean = self.FC_mean(x)
    log_var = self.FC_var(x)
    logging.debug("mean and var shapes:", mean.shape, log_var.shape)                                               #             (i.e., parateters of simple tractable normal distribution "q"
    return mean, log_var

class VAE_Decoder(nn.Module):
  def __init__(self, in_channels, hidden_dim, latent_dim):
        super(VAE_Decoder, self).__init__()
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                 out_channels=4,
                                 kernel_size=(1,8))
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=4,
                                 out_channels=8,
                                 kernel_size=(1,64))
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=8,
                                 out_channels=1,
                                 kernel_size=(1,32))
        
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.pool = nn.MaxPool2d((1, 2))
  def forward(self, x):
          #logging.debug("INPUT SHAPE", x.shape)
          h     = self.LeakyReLU(self.FC_hidden(x))
          #logging.debug("AFTER FC LAYER:", h.shape)
          h = h.view(h.shape[0], 1, 128, h.shape[1]//128)
          h = self._conv_trans_1(h)

          h = F.relu(h)
          h = self._conv_trans_2(h)
          #x = self.pool(x)
          h = F.relu(h)
          x = self._conv_trans_3(h)
          logging.debug("Final shape:", x.shape)
          return torch.sigmoid(x) # Make sure all values are between 0 and 1

class VAE_Model(nn.Module):
  def __init__(self, in_channels, hidden_dim, latent_dim):
    super(VAE_Model, self).__init__()
    self.Encoder = VAE_Encoder(in_channels, hidden_dim, latent_dim)
    self.Decoder = VAE_Decoder(in_channels, hidden_dim, latent_dim) 

  def reparameterization(self, mean, var):
    epsilon = torch.randn_like(var).to(DEVICE)
    z = mean + var * epsilon
    return z
  
  def forward(self, x):
    mean, log_var = self.Encoder(x)
    z = self.reparameterization(mean, torch.exp(0.5 * log_var))
    x_hat = self.Decoder(z)
    return x_hat, mean, log_var
