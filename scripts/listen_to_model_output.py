###########
# Imports #
###########
# From my other files:
from midi_utility import *
from vq_vae import * 

# General:
#from __future__ import print_function
#import matplotlib.pyplot as plt
#from scipy.signal import savgol_filter
#from six.moves import xrange
#import umap
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
#from torchvision.utils import make_grid
import pypianoroll

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


import os
from tqdm import tqdm
#import pandas as pd
#from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import random
import sys
from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL


modelpath = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\models\\'
datapath = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\new_data\\midi_tensors\\'
outpath = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\output_data\\'

num_hiddens = 128
embedding_dim = 32
commitment_cost = 0.5
num_embeddings = 64


def main():
    # Load model from memory
    model_dir = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\models\\model_10_25_2.pt'
    song_dir = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\data\\firstmodel_test\\'
    orig_npy = song_dir + 'Gimme! Gimme! Gimme!_0.npy'
    orig_midi = song_dir + "gimme_midi.mid"
    cropped_midi = song_dir + 'gimme_cropped.mid'

    '''data = midi_to_tensor(cropped_midi)
    #data = np.load(orig_npy)
    #tensor_to_midi(orig_tensor, orig_midi)
    #crop_midi(orig_midi, cropped_midi) #, maxlength=5)

    model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # Test on a song
    #data = np.load(datapath + 'Dancing Queen_1.npy')
    print(data.shape)
    p, n = data.shape

    l = 1024 # batch length

    data = data[:,:(data.shape[1]-(data.shape[1]%l))]
    p, n_2 = data.shape
    print("Cropped data shape:", data.shape)
    data = torch.tensor(data).float()

    chunked_data = data.view((n//l, 1, p, l))
    print("chunked data shape", chunked_data.shape)
    
    vq_loss, data_recon, perplexity = model(chunked_data)
    recon_error = F.mse_loss(data_recon, chunked_data) #/ data_variance
    loss = recon_error + vq_loss

    print("recon data shape:", data_recon.shape)
    for i in range(data_recon.shape[0]):
        print(torch.max(data_recon[i,:,:,:]).item())
    #print('Loss:', loss.item(), '\Perplexity:', perplexity)

    #chunked_data_np_array = chunked_data[:,:,:,10].detach().numpy()
    unchunked_recon = data_recon.view(p, n_2).detach().numpy()
    # Turn all negative values to 0 
    unchunked_recon = unchunked_recon.clip(min=0) # min note length that should count

    tensor_to_midi(unchunked_recon, song_dir + 'gimme_cropped_recon.mid')

    print("DONE")
    #data_recon_reshaped = data_recon.view(p,n_2)
    #tensor_to_midi(data_recon_reshaped, song_dir + 'gimme_recon.mid')

    #play_music(outpath + 'Dancing Queen_1_chunk_3_ORIGINAL.mid')
    #print("NEW")
    #play_music(song_dir + 'gimme_recon.mid')'''
    multitrack = pypianoroll.read(song_dir + 'gimme_cropped.mid')
    multitrack.plot()



if __name__ == "__main__":
    main()