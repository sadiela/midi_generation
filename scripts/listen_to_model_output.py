''' file for analyzing trained model results
- Reconstruct midis using model: reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0, norm=False)
- Save midi reconstructions: save_graphs(midi_path, save_path)
- Plot loss/perplexity for a model: show_result_graphs(yaml_name)
- Listen to midi reconstructions: play_music(midi_filename)
'''
import sys
sys.path.append("..") 
###########
# Imports #
###########
import sys
#sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import os
#print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
#print("PATH:", os.environ.get('PATH'))
# From my other files:
#from midi_utility import *
#from vq_vae import * 

# General:
#from __future__ import print_function
import matplotlib.pyplot as plt
#import pypianoroll
import yaml
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#import torch.optim as optim

import os
#from tqdm import tqdm
#import pandas as pd
#from skimage import io, transform
#import numpy as np
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

from statistics import mean, pstdev
import random
#import random
import sys
from pathlib import Path
#from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL

PROJECT_DIRECTORY = Path('..')


modelpath = PROJECT_DIRECTORY / 'models'
datapath = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'midi_tensors'
outpath = PROJECT_DIRECTORY / 'midi_data' / 'output_data'
respath = PROJECT_DIRECTORY / 'results'

#num_hiddens = 128
#embedding_dim = 128
#commitment_cost = 0.5
#num_embeddings = 1024
maxlength = 16*32
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_graphs(midi_path, save_path):
    print('saving pianoroll images/')
    file_list = os.listdir(midi_path)
    for file in tqdm(file_list):
        try:
            recon = pypianoroll.read(midi_path / file)
            recon.trim(0, 64*recon.resolution)
            recon.plot()
            plt.title(file)
            # FIX!
            plt.savefig(str(save_path / str(file.split('.')[0] + '.png')))
        except:
            print("passed", file)
        

def reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0, norm=False):
    res_string = "RECON ERRORS!\n"
    file_list = os.listdir(orig_tensor_dir)
    for file in tqdm(file_list):
        cur_tensor, loss, recon_err, zero_recon = reconstruct_song(orig_tensor_dir / file, model_path, clip_val=clip_val, norm=norm)
        res_string += str(file) + ' recon error: ' + str(recon_err.item()) + ' loss: ' + str(loss.item()) + ' zero recon:' + str(zero_recon.item()) + '\n'
        # save tensor
        np.save(new_tensor_dir / str(file.split('.')[0] + '_conv.npy'), cur_tensor)
        # convert to midi and save midi 
        tensor_to_midi(cur_tensor, new_midi_dir / str(file.split('.')[0] + '.mid'))
        #input("continue...")
    with open(new_midi_dir / 'recon_info.txt', 'w') as outfile:
        outfile.write(res_string)

def reconstruct_song(orig_tensor_path, model_path, clip_val=0, norm=False):
    data = np.load(orig_tensor_path)
    if norm:
        data = data / maxlength
    
    model = Model(num_embeddings=1024, embedding_dim=128, commitment_cost=0.5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Test on a song
    print(data.shape)
    p, n = data.shape

    l = 128 #1024 # batch length

    data = data[:,:(data.shape[1]-(data.shape[1]%l))]
    p, n_2 = data.shape
    print("Cropped data shape:", data.shape)
    data = torch.tensor(data).float()

    chunked_data = data.view((n//l, 1, p, l))
    print("chunked data shape", chunked_data.shape)
    print(data)
    
    vq_loss, data_recon, perplexity = model(chunked_data)
    recon_error = F.mse_loss(data_recon, chunked_data) #/ data_variance
    zero_recon = F.mse_loss(torch.zeros(n//l, 1, p, l), chunked_data)
    loss = recon_error + vq_loss

    print("recon data shape:", data_recon.shape)
    print(data_recon)
    for i in range(data_recon.shape[0]):
        print(torch.max(data_recon[i,:,:,:]).item())
    print('Loss:', loss.item(), '\Perplexity:', perplexity.item())

    unchunked_recon = data_recon.view(p, n_2).detach().numpy()
    # Turn all negative values to 0 
    unchunked_recon = unchunked_recon.clip(min=clip_val) # min note length that should count

    if norm: # unnormalize!
        unchunked_recon = unchunked_recon * maxlength
    #tensor_to_midi(unchunked_recon, new_midi_path)

    return unchunked_recon, loss, recon_error, zero_recon

def show_result_graphs(yaml_dir, yaml_name, plot_dir):
    root_name = yaml_name.split(".")[0]
    with open(yaml_dir / yaml_name) as file: 
        res_dic = yaml.load(file, Loader=yaml.FullLoader)
    plt.plot(res_dic['reconstruction_error'])
    plt.title("Reconstruction Error" + root_name)
    plt.xlabel("Iteration")
    #plt.show()
    print("SAVING")
    plt.savefig(str(plot_dir / str(root_name+".png")))

    plt.clf()

def main():

    # Load model from memory
    ### MODELS ###
    MODEL_DIRECTORY = PROJECT_DIRECTORY / "models"
    YAML_DIRECTORY = PROJECT_DIRECTORY / "results" / "l1_reg_test_yamls"
    PLOT_DIRECTORY = PROJECT_DIRECTORY / "results" / "l1_reg_test_error_plots"

    yaml_list = os.listdir(YAML_DIRECTORY)
    for yaml_file in yaml_list:
        print(yaml_file)
        show_result_graphs(YAML_DIRECTORY, yaml_file, PLOT_DIRECTORY)

    print("DONE")

    '''
    mse_model_path = PROJECT_DIRECTORY / 'models' / 'model_mse-2021-11-282.pt'
    #mae_model_path = PROJECT_DIRECTORY / 'models' / 'model_mae-2021-11-280.pt'
    #msenorm_model_path = PROJECT_DIRECTORY / 'models' / 'model_msenorm-2021-11-280.pt'
    #maenorm_model_path = PROJECT_DIRECTORY / 'models' / 'model_maenorm-2021-11-280.pt'

    ### TENSORS ###
    orig_tensor = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'originals'
    mse_tensor = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'mse_tensor'
    #mae_tensor = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'mae_tensor'
    #msenorm_tensor = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'msenorm_tensor'
    #maenorm_tensor = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'maenorm_tensor'

    ### MIDIS ###
    old_midi = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'old_midi'
    mse_midi = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'mse_midi'
    #mae_midi = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'mae_midi'
    #msenorm_midi = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'msenorm_midi'
    #maenorm_midi = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'listening_test' / 'maenorm_midi'

    ### PIANOROLLS ###
    orig_pianorolls = PROJECT_DIRECTORY / 'results' / 'original_midi_pianorolls'
    mse_res = PROJECT_DIRECTORY / 'results' / 'mse_midi_pianorolls'
    #mae_res = PROJECT_DIRECTORY / 'results' / 'mae_midi_pianorolls'
    #msenorm_res = PROJECT_DIRECTORY / 'results' / 'msenorm_midi_pianorolls'
    #maenorm_res = PROJECT_DIRECTORY / 'results' / 'maenorm_midi_pianorolls'

    # Save original pianoroll images
    print("Saving og pianorolls")
    #save_graphs(old_midi, orig_pianorolls)

    # Reconstruct songs in accordance to each model
    print("Reconstructing")
    reconstruct_songs(orig_tensor, mse_tensor, mse_midi, mse_model_path, clip_val=0)
    '''print("Reconstructing")
    reconstruct_songs(orig_tensor, mae_tensor, mae_midi, mae_model_path, clip_val=0)
    print("Reconstructing")
    reconstruct_songs(orig_tensor, msenorm_tensor, msenorm_midi, msenorm_model_path, clip_val=0, norm=True)
    print("Reconstructing")
    reconstruct_songs(orig_tensor, maenorm_tensor, maenorm_midi, maenorm_model_path, clip_val=0, norm=True)
    '''

    '''
    # save midis for each reconstruction
    print("Saving new pianorolls")
    #save_graphs(mse_midi, mse_res)
    #save_graphs(mae_midi, mae_res)
    #save_graphs(msenorm_midi, msenorm_res)
    #save_graphs(maenorm_midi, maenorm_res)

    print("DONE!")
    results={"reconstruction_error": recon_error, "perplexity": perplex, "nan_reconstruction_files": nan_recon_files}
    savefile = get_free_filename('results_' + fstub, resultspath, suffix='.yaml')
    print("SAVING FILE TO:", savefile)
    with open(savefile, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    #orig_npy = song_dir + 'Gimme! Gimme! Gimme!_0.npy'
    #orig_midi = outputs + "gimme_midi.mid"
    #cropped_midi = outputs + 'gimme_cropped.mid'

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
    #yaml_name1 = Path('../results/results_mae-2021-11-280.yaml')
    #yaml_name2 = Path('../results/results_mse-2021-11-280.yaml')
    #yaml_name3 = Path('../results/results_maenorm-2021-11-280.yaml')
    #yaml_name4 = Path('../results/results_msenorm-2021-11-290.yaml')

    #show_result_graphs(yaml_name1)
    #show_result_graphs(yaml_name2)
    #show_result_graphs(yaml_name3)
    #show_result_graphs(yaml_name4)


if __name__ == "__main__":
    # main()
    #respath = r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\results\\L1_REG_TESTS\\results_l1reglosstest000_1-2022-04-17-0.yaml'
    #show_result_graphs(respath)
    #respath = r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\results\\L1_REG_TESTS\\results_l1reglosstest000_5-2022-04-17-0.yaml'
    #show_result_graphs(respath)
    #respath = r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\results\\L1_REG_TESTS\\results_l1reglosstest00_1-2022-04-17-0.yaml'
    #show_result_graphs(respath)

    model_path = Path(r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\models\\new_l1_test_models\\model_mse_test-2022-05-08-0.pt')
    #res_folder = Path(r'C:\\Users\\sadie\\Documents\BU\\fall_2021\\research\\music\\midi_data\\new_data\\listening_test\\L1_reg_small_lambda')
    res_folder = Path(r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\results\\L1_result_reconstructions')
    orig_tensor = Path(r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\new_data\\listening_test\\originals')
    reconstruct_songs(orig_tensor, res_folder, res_folder, model_path, clip_val=0)
    save_graphs(res_folder, res_folder)

    '''
    midi_path = r'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\new_data\\listening_test\\L1reg_smaller\\AllMyLoving_2_cropped.mid'
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    try: 
        print(midi_data.estimate_tempo())
    except Exception as e:
        print(e)
    for instrument in midi_data.instruments:
        print(instrument.notes)
    '''
    #recon = pypianoroll.read()
    #recon.trim(0, 64*recon.resolution)
    #recon.plot()
    