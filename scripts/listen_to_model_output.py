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
    midifile = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\data\\single_track_midis\\Money, Money, Money_0.mid'
    #play_music(midifile)
    cropped_midifile_path = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\scripts\\cropped_money.mid'
    crop_midi(midifile, cropped_midifile_path, maxlength=5)
    print("CROPPED")
    play_music(cropped_midifile_path)
    '''model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    model.load_state_dict(torch.load(modelpath + 'model_10_25_2.pt'))
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
    play_music(outpath + 'Dancing Queen_1_chunk_3.mid')'''

if __name__ == "__main__":
    main()