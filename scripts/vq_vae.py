'''
This file contains the VQ-VAE model class
'''
###########
# Imports #
###########
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random 
import os
from tqdm import tqdm
import pickle
import logging
#from dp_loss import *
from dp_loss.sparse_dp_loss import *
from pathlib import Path
#from midi_utility import * 


# is reconstruction error going down? 
# Run w/ more data
# Increase # of embedding vectors? 
# shuffle chunks? 

##############################
# MODEL/OPTIMIZER PARAMETERS #
##############################
'''num_training_updates = 15000
num_hiddens = 128
num_residual_hiddens = 16
num_residual_layers = 2
l = 512 #1024 # batch length
<<<<<<< HEAD
=======
p= 36 #128
>>>>>>> d34803d735ac971341aaa5e7cd75d61126b9cc15
p= 36 #128'''
decay = 0.99
learning_rate = 1e-3
#num_embeddings = 64
#embedding_dim = 128
#commitment_cost = 0.5

#####################
# CUSTOM DATALOADER #
#####################
class MidiDataset(Dataset):
    """Midi dataset."""

    def __init__(self, npy_file_dir, l=512, sparse=False, norm=False):
        """
        Args:
            npy_file_dir (string): Path to the npy file directory
        """
        file_list = os.listdir(npy_file_dir)
        self.l = l
        self.norm = norm 
        self.maxlength = 16*32
        self.sparse = sparse
        self.paths = [ Path(npy_file_dir) / file for file in file_list] # get entire list of midi tensor file names 
        
        #self.batch_file_paths = set()

    def __getitem__(self, index):
        # choose random file path from directory (not already chosen), chunk it 
        print(str(self.paths[index]))
        # load in tensor
        if self.sparse:
          with open(self.paths[index], 'rb') as f:
            pickled_tensor = pickle.load(f)
          cur_tensor = pickled_tensor.toarray()
        else:
          cur_tensor = np.load(self.paths[index]) #, allow_pickle=True)

        # convert to torch tensor (vs numpy tensor)
        cur_data = torch.tensor(cur_tensor)
        #cur_data = cur_data[46:-46,:]
        p, l_i = cur_data.shape
        
        # normalize if specified
        if self.norm:
          cur_data = cur_data / self.maxlength 
        
        # make sure divisible by l
        # CHUNK! 
        #print("DATA SHAPE:", cur_data.shape)
        if l_i // self.l == 0: 
          padded = torch.zeros((p, self.l))
          padded[:,0:l_i] = cur_data
          l_i=self.l
        else: 
          padded = cur_data[:,:(cur_data.shape[1]-(cur_data.shape[1]%self.l))]
        padded = padded.float()
        cur_chunked = torch.reshape(padded, (l_i//self.l, 1, p, self.l)) 
        
        return cur_chunked # 3d tensor: l_i\\l x p x l

    def __getname__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

# MODEL CLASS DEFINITIONS #
# input: p x t, t variable! p=128
class MIDIVectorQuantizer(nn.Module):
  def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_cost=0.5):
    super().__init__()

    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost

    self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
    self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings) # randomize embeddings to start

  def forward(self, inputs):
    logging.info("MIDI VECTOR QUANTIZER FORWARD PASS")
    # PASS ONE SONG AT A TIME
    # inputting convolved midi tensors
    # batch dependent on song length, train one song at a time 
    # input = b x p x l

    logging.debug("Original input shape: %s", str(inputs.shape))
    inputs = inputs.squeeze(1) # need to be 2d
    
    # we will embed along dim p 
    inputs = inputs.permute(0,2,1).contiguous() # now bxlxp
    # flatten input
    input_shape = inputs.shape 
    flat_input = inputs.view(-1, self._embedding_dim)
    #(bxl)xp
    logging.debug(flat_input, flat_input.shape)

    distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
    logging.debug("DISTANCES: %s %s", str(distances), str(distances.shape))

    # Encoding
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    #print(encoding_indices)
    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1)
    #print("ENCODINGS:", encodings)
    
    # Quantize and unflatten
    quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
    logging.info(str(quantized) + str(quantized.shape))

     # Loss
    e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    # e_latent_loss = F.l1_loss(quantized.detach(), inputs)
    q_latent_loss = F.mse_loss(quantized, inputs.detach())
    # q_latent_loss = F.l1_loss(quantized, inputs.detach())
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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, quantize=True, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(1)

        self._vq_vae = MIDIVectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self._decoder = Decoder(embedding_dim)

        self.quantize = quantize

    def forward(self, x):
      #print("INPUT DIMENSION", x.shape)
      #input('Continue...')
      if not self.quantize:
        z = self._encoder(x)
        x_recon = self._decoder(z)
        return 0, x_recon, 0
      else: 
        z = self._encoder(x)
        if self.quantize: 
          loss, quantized, perplexity, _ = self._vq_vae(z)
          x_recon = self._decoder(quantized)

          return loss, x_recon, perplexity
        else:
          x_recon = self._decoder(z)
          return 0, x_recon, 0

def collate_fn(data, collate_shuffle=True):
  # data is a list of tensors
  # concatenate and shuffle all list items
  full_list = torch.cat(data, 0)
  if collate_shuffle:
    idx = torch.randperm(full_list.shape[0])
    return  full_list[idx].view(full_list.size())
  else:
    return full_list

def train_model(datapath, model, save_path, learning_rate=learning_rate, lossfunc='mse', bs=10, batchlength=256, normalize=False, quantize=True, sparse=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midi_tensor_dataset = MidiDataset(datapath, l=batchlength, norm=normalize, sparse=sparse)

    # declare model and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.float()
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    total_loss = []
    nanfiles = []

    training_data = DataLoader(midi_tensor_dataset, collate_fn=collate_fn, batch_size=bs, shuffle=True, num_workers=2)

      # Let # of tensors = n
      # each tensor is pxl_i, where l_i is the length of the nth tensor
      # when we chunk the data, it becomes (l_i//l = s_i) x 1 x p x l 
      # so we want a big (sum(s_i)) x 1 x p x l tensor. 
      # Then we want to shuffle along axis=0 so two adjacent pxl guys aren't 
      # necessarily from the same song

    logging.info("Device: %s" , device)
    max_tensor_size= 0 

    dynamic_loss = SparseDynamicLoss.apply
    lam = 5

    for i, data in tqdm(enumerate(training_data)):
        #name = midi_tensor_dataset.__getname__(i)
        # s x p x 1 x l
        data = data.to(device)
        cursize = torch.numel(data)
        if cursize > max_tensor_size:
          max_tensor_size = cursize
          logging.info("NEW MAX BATCH SIZE: %d", max_tensor_size)

        print('TRAIN:', data.shape)

        '''with profile(
          activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
          with_stack=True,
        ) as prof:'''
        vq_loss, data_recon, perplexity = model(data)
        if lossfunc=='mse':
          recon_error = F.mse_loss(data_recon, data) #/ data_variance
        elif lossfunc=='dyn':
<<<<<<< HEAD
          recon_error = dynamic_loss(data_recon, data, device) #X_hat, then X!!!
=======
          print("ENTERING LOSS!", i)
          recon_error = dynamic_loss(data_recon, data, device) #X_hat, then X!!!
        elif lossfunc=='l1reg':
          recon_error = F.mse_loss(data_recon, data) + lam*torch.norm(data_recon, p=1) # +  ADD L1 norm
>>>>>>> d34803d735ac971341aaa5e7cd75d61126b9cc15
        else: # loss function = mae
          recon_error = F.l1_loss(data_recon, data)
        loss = recon_error + vq_loss # will be 0 if no quantization
        loss.backward()
        #print("backpropagated")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        #output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        #print(output) 
        
        total_loss.append(loss.item())
        if quantize:
          train_res_recon_error.append(recon_error.item())
          train_res_perplexity.append(perplexity.item())
        else:
          train_res_recon_error.append(loss)
          train_res_perplexity.append(perplexity)

        if pd.isna(recon_error.item()):
          nanfiles.append(midi_tensor_dataset.__getname__(i))

        if (i+1) % 100 == 0:
          torch.save({
                      'iteration': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': train_res_recon_error[-1],
                      }, save_path)
          logging.info('%d iterations' % (i+1))
          logging.info('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
          logging.info('\n')

    logging.info("saving model to %s"%save_path)
    torch.save(model.state_dict(), save_path)
    return train_res_recon_error, train_res_perplexity, nanfiles
