###########
# Imports #
###########
# From my other files:
from midi_utility import *
from vq_vae import * 

# General:
#from __future__ import print_function
import matplotlib.pyplot as plt
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


modelpath = PROJECT_DIRECTORY + 'models\\'
datapath = PROJECT_DIRECTORY + 'midi_data\\new_data\\midi_tensors\\'
outpath = PROJECT_DIRECTORY + 'midi_data\\output_data\\'

num_hiddens = 128
embedding_dim = 32
commitment_cost = 0.5
num_embeddings = 64

def reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0):
    file_list = os.listdir(tensor_dir)
    for file in tqdm(file_list):
        cur_tensor = reconstruct_song(orig_tensor_dir + '\\' + file, model_path, clip_val=clip_val)
        # save tensor
        np.save(new_tensor_dir + file.split('.')[0] + '_conv.npy', cur_tensor)
        # convert to midi and save midi 
        tensor_to_midi(cur_tensor, new_midi_dir + '\\' + file.split('.')[0] + '.mid')

def reconstruct_song(orig_tensor_path, model_path, clip_val=0):
    data = np.load(orig_tensor_path)
    
    model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test on a song
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
    print('Loss:', loss.item(), '\Perplexity:', perplexity.item())

    #chunked_data_np_array = chunked_data[:,:,:,10].detach().numpy()
    unchunked_recon = data_recon.view(p, n_2).detach().numpy()
    # Turn all negative values to 0 
    unchunked_recon = unchunked_recon.clip(min=clip_val) # min note length that should count

    #tensor_to_midi(unchunked_recon, new_midi_path)

    return unchunked_recon

def main():
    # Load model from memory
    model_dir = PROJECT_DIRECTORY + 'models\\model_10_25_2.pt'
    song_dir = PROJECT_DIRECTORY + 'midi_data\\new_data\\midi_tensors\\'
    outputs = PROJECT_DIRECTORY + 'midi_data\\new_data\\listening_test\\'
    orig_npy = song_dir + 'Gimme! Gimme! Gimme!_0.npy'
    orig_midi = outputs + "gimme_midi.mid"
    cropped_midi = outputs + 'gimme_cropped.mid'
    
    recon = pypianoroll.read(PROJECT_DIRECTORY + 'midi_data\\single_track_midis\\Eagle_1.mid')
    recon.plot()
    plt.show()
    play_music(PROJECT_DIRECTORY + 'midi_data\\single_track_midis\\Eagle_1.mid')

    # loop through midi tensors/print max value in all midi tensors ... are there nans? where? 
    file_list = os.listdir(song_dir)
    for file in file_list:
        cur_tensor = np.load(song_dir + '\\' + file)
        if cur_tensor.max() > 1000: 
            print(file, cur_tensor.max()) # plot a histogram of these 
    print("done")
        

    #orig_tensor = np.load(orig_npy)
    #tensor_to_midi(orig_tensor, orig_midi)
    #crop_midi(orig_midi, cropped_midi) #, maxlength=5)
    #reconstruct_song(orig_npy, outputs + 'recon_2.mid', model_dir, clip_val=0.01)

    #play_music(outpath + 'Dancing Queen_1_chunk_3_ORIGINAL.mid')
    #print("NEW")
    #play_music(outputs + 'gimme_cropped_recon.mid')
    #multitrack = pypianoroll.read(outputs + 'gimme_cropped.mid')
    #multitrack.plot()
    #recon = pypianoroll.read(outputs + 'recon_2.mid')
    #recon.plot()
    #plt.show()


if __name__ == "__main__":
    main()