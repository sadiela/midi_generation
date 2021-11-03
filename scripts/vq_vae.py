'''
This file contains the initial VQ-VAE model clas
'''
###########
# Imports #
###########
#from __future__ import print_function

#import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
#from six.moves import xrange
#import umap
#import pandas as pd
#from skimage import io, transform
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
#from torchvision.utils import make_grid
#from torchvision import transforms, utils

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm

# is reconstruction error going down? 
# Run w/ more data
# Increase # of embedding vectors? 
# shuffle chunks? 

##############################
# MODEL/OPTIMIZER PARAMETERS #
##############################
num_training_updates = 15000
num_hiddens = 128
num_residual_hiddens = 16
num_residual_layers = 2
l = 1024 # batch length
decay = 0.99
learning_rate = 1e-3
num_embeddings = 64
embedding_dim = 128
commitment_cost = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################
# CUSTOM DATALOADER #
#####################
class MidiDataset(Dataset):
    """Midi dataset."""

    def __init__(self, npy_file_dir):
        """
        Args:
            npy_file_dir (string): Path to the npy file directory
        """
        self.midi_tensors = []
        file_list = os.listdir(npy_file_dir)
        for file in tqdm(file_list): 
          #print(npy_file_dir + file)
          cur_tensor = np.load(npy_file_dir + file) #, allow_pickle=True)
          self.midi_tensors.append((file,cur_tensor)) # each one is a tuple now
        #self.root_dir = root_dir
        #self.transform = transform

    def __getitem__(self, index):
        return self.midi_tensors[index][1]

    def __getname__(self, index):
        return self.midi_tensors[index][0]

    def __len__(self):
        return len(self.midi_tensors)

# MODEL CLASS DEFINITIONS #
# input: p x t, t variable! p=128
class MIDIVectorQuantizer(nn.Module):
  def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_cost=0.5):
    super().__init__()

    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost

    self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
    self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

  def forward(self, inputs):
    # PASS ONE SONG AT A TIME
    # inputting convolved midi tensors
    # batch dependent on song length, train one song at a time 
    # input = b x p x l

    inputs = inputs.squeeze(1)
    #print(inputs.shape)
    
    # we will embed along dim p 
    inputs = inputs.permute(0,2,1).contiguous() # now bxlxp
    # flatten input
    input_shape = inputs.shape 
    flat_input = inputs.view(-1, self._embedding_dim)
    #(bxl)xp
    #print(flat_input, flat_input.shape)

    distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
    #print("DISTANCES", distances, distances.shape)

    # Encoding
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    #print(encoding_indices)
    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1)
    #print("ENCODINGS:", encodings)
    
    # Quantize and unflatten
    quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
    #print(quantized, quantized.shape)

     # Loss
    e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    q_latent_loss = F.mse_loss(quantized, inputs.detach())
    loss = q_latent_loss + self._commitment_cost * e_latent_loss
    
    quantized = inputs + (quantized - inputs).detach() # backprop through delta
    avg_probs = torch.mean(encodings, dim=0)
    # make sure embeddings are far from each other 
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    
    # convert quantized from 
    return loss, quantized.permute(0, 2, 1).contiguous().unsqueeze(1), perplexity, encodings

class Encoder(nn.Module):
  def __init__(self, in_channels):
        super(Encoder, self).__init__()
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
  def forward(self, inputs):
          #print(inputs.shape)
          x = self._conv_1(inputs)
          #print(x.shape)
          #x = self.pool(x)
          #print(x.shape)
          x = F.relu(x)
          #print(x.shape)

          x = self._conv_2(x)
          #print(x.shape)
          #x = self.pool(x)
          #print(x.shape)
          x = F.relu(x)
          #print(x.shape)
          x = self._conv_3(x)
          #print(x.shape)
          return x

class Decoder(nn.Module):
  def __init__(self, in_channels=1):
        super(Decoder, self).__init__()
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=1,
                                 out_channels=4,
                                 kernel_size=(1,8))
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=4,
                                 out_channels=8,
                                 kernel_size=(1,64))
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=8,
                                 out_channels=1,
                                 kernel_size=(1,32))
        self.pool = nn.MaxPool2d((1, 2))
  def forward(self, inputs):
          #print(inputs.shape)
          x = self._conv_trans_1(inputs)
          #print(x.shape)
          #x = self.pool(x)
          #print(x.shape)
          x = F.relu(x)
          #print(x.shape)

          x = self._conv_trans_2(x)
          #print(x.shape)
          #x = self.pool(x)
          #print(x.shape)
          x = F.relu(x)
          #print(x.shape)
          x = self._conv_trans_3(x)
          #print(x.shape)
          return x

class Model(nn.Module):
    def __init__(self, num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(1)

        self._vq_vae = MIDIVectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self._decoder = Decoder(embedding_dim)

    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


def train_model(datapath, model, save_path, learning_rate=learning_rate):
    midi_tensor_dataset = MidiDataset(datapath)
    # declare model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.float()
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    nanfiles = []

    for i in tqdm(range(midi_tensor_dataset.__len__())):
        #name = midi_tensor_dataset.__getname__(i)
        data = midi_tensor_dataset.__getitem__(i)
        p, n = data.shape
        data = torch.tensor(data)

        data = data[:,:(data.shape[1]-(data.shape[1]%l))]
        data = data.float()
        print(data.size())

        chunked_data = torch.reshape(data, (n//l, 1, p, l))
        chunked_data = chunked_data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(chunked_data)
        recon_error = F.mse_loss(data_recon, chunked_data) #/ data_variance
        loss = recon_error + vq_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if pd.isna(recon_error.item()):
          nanfiles.append(midi_tensor_dataset.__getname__(i))

        if (i+1) % 10 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-10:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-10:]))
            print()

    torch.save(model.state_dict(), save_path)
    return train_res_recon_error, train_res_perplexity, nanfiles