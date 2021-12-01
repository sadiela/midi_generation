###########
# Imports #
###########
# From my other files:
from midi_utility import *
from vq_vae import * 

# General:
#from __future__ import print_function
import matplotlib.pyplot as plt
import pypianoroll
import yaml
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
from pathlib import Path
#from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL


modelpath = PROJECT_DIRECTORY / 'models'
datapath = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'midi_tensors'
outpath = PROJECT_DIRECTORY / 'midi_data' / 'output_data'

num_hiddens = 128
embedding_dim = 128
commitment_cost = 0.5
num_embeddings = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0):
    file_list = os.listdir(orig_tensor_dir)
    for file in tqdm(file_list):
        cur_tensor = reconstruct_song(orig_tensor_dir + '\\' + file, model_path, clip_val=clip_val)
        # save tensor
        np.save(new_tensor_dir + file.split('.')[0] + '_conv.npy', cur_tensor)
        # convert to midi and save midi 
        tensor_to_midi(cur_tensor, new_midi_dir + '\\' + file.split('.')[0] + '.mid')

def reconstruct_song(orig_tensor_path, model_path, clip_val=0):
    data = np.load(orig_tensor_path)
    
    model = Model(num_embeddings=1024, embedding_dim=128, commitment_cost=commitment_cost)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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

    unchunked_recon = data_recon.view(p, n_2).detach().numpy()
    # Turn all negative values to 0 
    unchunked_recon = unchunked_recon.clip(min=clip_val) # min note length that should count

    #tensor_to_midi(unchunked_recon, new_midi_path)

    return unchunked_recon

def show_result_graphs(yaml_name):
    with open(yaml_name) as file: 
        res_dic = yaml.load(file) #, Loader=yaml.FullLoader)
    plt.plot(res_dic['reconstruction_error'])
    plt.title("Reconstruction Error")
    plt.xlabel("Iteration")
    plt.show()

    plt.plot(res_dic['perplexity'])
    plt.title("Perplexity")
    plt.xlabel("Iteration")
    plt.show()

def main():
    # Load model from memory
    model_path = PROJECT_DIRECTORY + 'models\\model-14.pt'
    orig_tensor_dir = PROJECT_DIRECTORY + 'midi_data\\new_data\\listening_test\\originals\\'
    new_tensor_dir = PROJECT_DIRECTORY + 'midi_data\\new_data\\listening_test\\new\\'
    new_midi_dir =  PROJECT_DIRECTORY + 'midi_data\\new_data\\listening_test\\new_midi\\'
    old_midi_dir = PROJECT_DIRECTORY + 'midi_data\\new_data\\listening_test\\old_midi\\'
    #orig_npy = song_dir + 'Gimme! Gimme! Gimme!_0.npy'
    #orig_midi = outputs + "gimme_midi.mid"
    #cropped_midi = outputs + 'gimme_cropped.mid'
    #reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0)

    #tensors_to_midis(orig_tensor_dir, old_midi_dir)
    
    '''print("PLOT")
    file_list = os.listdir(old_midi_dir)
    for file in file_list:
        recon = pypianoroll.read(old_midi_dir + file)
        try: 
            recon.trim(0, 64*recon.resolution)
        except:
            print("passed")
        recon.plot()
        plt.title(file)
        plt.show()

    print("RECONSTRUCTED:")
    file_list = os.listdir(new_midi_dir)
    for file in file_list:
        try:
            recon = pypianoroll.read(new_midi_dir + file)
        except:
            pass
        try:
            recon.trim(0, 64*recon.resolution)
        except:
            print("passed", file)
        recon.plot()
        plt.title(file)
        plt.show()'''

    '''print("S")
    play_music(new_midi_dir + 'Andante,Andante_8_cropped.mid')
    print("DONE")
    '''

    # loop through midi tensors/print max value in all midi tensors ... are there nans? where? 
    '''file_list = os.listdir(orig_tensor_dir)
    for file in file_list:
        cur_tensor = np.load(orig_tensor_dir + '\\' + file)
        if cur_tensor.max() > 1000: 
            print(file, cur_tensor.max()) # plot a histogram of these 
    print("done")'''
        

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
    
    # PLOT RESULTS 
    yaml_name = Path('../results/results_all-0.yaml')
    show_result_graphs(yaml_name)


if __name__ == "__main__":
    main()