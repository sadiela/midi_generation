###########
# Imports #
###########
from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import os
from tqdm import tqdm
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import random
import sys
from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL
import os
import pygame
import json
import numpy as np
import pretty_midi

modelpath = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\models\\'
datapath = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\new_data\\midi_tensors\\'
outpath = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\output_data\\'

num_hiddens = 128
embedding_dim = 32
commitment_cost = 0.5
num_embeddings = 64



class MidiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, npy_file_dir):
        """
        Args:
            npy_file_dir (string): Path to the npy file directory
        """
        self.midi_tensors = []
        file_list = os.listdir(npy_file_dir)
        for file in tqdm(file_list): 
          print(npy_file_dir + file)
          cur_tensor = np.load(npy_file_dir + file)
          self.midi_tensors.append(cur_tensor) 
        #self.root_dir = root_dir
        #self.transform = transform

    def __getitem__(self, index):
        return self.midi_tensors[index]

    def __len__(self):
        return len(self.midi_tensors)

#### VARIABLE DEFINITIONS
# n = original song length
# m = length after encoding layer
# l = length of batch
# b = batch size (VARIABLE) NUMBER OF CHUNKS IN ONE SONG
# k = number of embeddings
# p = pitch dimension AND embedding dimension

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
  def __init__(self, in_channels, num_hidden):
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
  def __init__(self, in_channels=1, num_hidden=1):
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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(1, num_hiddens)

        self._vq_vae = MIDIVectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


def tensor_to_midi(tensor, desired_filepath):
    # Converts midi tensor back into midi file format
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - tempo = 120bpm
    #   - smallest note subdivision = eighth note (0.250 seconds)
    #   - writes everything as piano 
    # Create new midi object
    new_mid = pretty_midi.PrettyMIDI() # type=0
    # create a track and add it to the midi
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    for time in range(tensor.shape[1]):
        for pitch in range(tensor.shape[0]):
            if tensor[pitch,time,0] != 0:
                print("NOTE DETECTED")
                new_note = pretty_midi.Note(velocity=100, pitch=(pitch), start=(time/4), end=((time/4)+(tensor[pitch,time,0]/4)))
                piano.notes.append(new_note)
    new_mid.instruments.append(piano)
        # save to .mid file 
    new_mid.write(desired_filepath)

# PLAYBACK CONFIGURATION
# mixer config
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)
pygame.mixer.music.set_volume(0.8) # optional volume 0 to 1.0

def play_music(midi_filename):
    try:
        # use the midi file you just saved
        #Stream music_file in a blocking manner
        clock = pygame.time.Clock()
        pygame.mixer.music.load(midi_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30) # check if playback has finished
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

def main():
    # Load model from memory
    model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    model.load_state_dict(torch.load(modelpath + 'model_10_25.pt'))
    model.eval()

    # Test on a song
    data = np.load(datapath + 'Dancing Queen_1.npy')
    print(data.shape)
    p, n = data.shape

    l = 1024 # batch length

    data = data[:,:(data.shape[1]-(data.shape[1]%l))]
    data = torch.tensor(data).float()

    chunked_data = data.view((n//l, 1, p, l))
    
    vq_loss, data_recon, perplexity = model(chunked_data)

    print('Loss:', vq_loss, '\Perplexity:', perplexity)

    chunked_data_np_array = chunked_data[:,:,:,10].detach().numpy()
    tensor_to_midi(chunked_data_np_array, outpath + 'Dancing Queen_1_chunk_3_ORIGINAL.mid')

    one_chunk = data_recon[:,:,:,10].detach().numpy() #torch.squeeze(data_recon[:,:,:,3], 1)
    tensor_to_midi(one_chunk, outpath + 'Dancing Queen_1_chunk_3.mid')

    play_music(outpath + 'Dancing Queen_1_chunk_3_ORIGINAL.mid')
    print("NEW")
    play_music(outpath + 'Dancing Queen_1_chunk_3.mid')

if __name__ == "__main__":
    main()