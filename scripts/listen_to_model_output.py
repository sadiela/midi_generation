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
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn.functional as F
from midi_utility import *
from vq_vae import * 
import matplotlib.pyplot as plt
import pypianoroll
import yaml
from pathlib import Path
import argparse
import pickle

maxlength = 16*32

def save_graphs(midi_path, save_path):
    print('saving pianoroll images')
    file_list = os.listdir(midi_path)
    for file in tqdm(file_list):
        try:
            recon = pypianoroll.read(Path(midi_path) / file)
            print(recon.get_length())
            if recon.get_length() > 64*recon.resolution: # trim only if long
                recon.trim(0, 64*recon.resolution)
            recon.plot()
            plt.title(file)
            # FIX!
            plt.savefig(str(Path(save_path) / str(file.split('.')[0] + '.png')))
        except Exception as e:
            print(e)
            print("passed", file)
        
def reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0, norm=False, batchlength=256, num_embed=1024):
    res_string = "MODEL FILE NAME" + str(model_path) + "\nRECON ERRORS!\n"
    file_list = os.listdir(orig_tensor_dir)

    print("declaring model")
    model = Model(num_embeddings=num_embed, embedding_dim=128, commitment_cost=0.5)
    print(sum(p.numel() for p in model.parameters()))
    print("loading saved model")
    stat_dictionary = torch.load(model_path, map_location=torch.device('cpu'))
    model_params = stat_dictionary["model_state_dict"]
    print("putting params in model")
    print(sum(p.numel() for p in model_params.values()))
    model.load_state_dict(model_params, )
    print("setting to eval mode")
    model.eval()

    print("listing files")
    for file in file_list:
        print(file) # perform reconstruction
        cur_tensor, loss, recon_err, zero_recon = reconstruct_song(Path(orig_tensor_dir) / file, model, clip_val=clip_val, norm=norm, batchlength=batchlength)
        # record info IF RECONSTRUCTION NOT ALL 0s
        if (cur_tensor > 0).sum() > 0: 
            print(cur_tensor[:,:10])
            #input("Continue")
            res_string += str(file) + ' recon error: ' + str(recon_err.item()) + ' loss: ' + str(loss.item()) + ' zero recon:' + str(zero_recon.item()) + '\n'
            # save tensor
            sparse_arr = sparse.csr_matrix(cur_tensor) # save sparse!!!
            with open(str(Path(new_tensor_dir) / str(file.split('.')[0] + '_conv.p')), 'wb') as outfile:
                pickle.dump(sparse_arr, outfile)
            # convert to midi and save midi 
            print("entering tensor to midi")
            tensor_to_midi_2(cur_tensor, Path(new_midi_dir) / str(file.split('.')[0] + '.mid'), pitchlength_cutoff=0.2)
        else:
            print(file, "reconstruction is all 0s")
    with open(Path(new_midi_dir) / 'recon_info.txt', 'w') as outfile:
        outfile.write(res_string)

def reconstruct_song(orig_tensor_path, model, clip_val=0, norm=False, batchlength=256):
    with open(orig_tensor_path,'rb') as f: 
        pickled_tensor = pickle.load(f)
    data = pickled_tensor.toarray()
    if norm:
        data = data / maxlength

    # Test on a song
    #print(data.shape)
    p, n = data.shape

    l = batchlength #1024 # batch length

    data = data[:,:(data.shape[1]-(data.shape[1]%l))]
    p, n_2 = data.shape
    #print("Cropped data shape:", data.shape)
    data = torch.tensor(data).float()

    chunked_data = data.view((n//l, 1, p, l))
    #print("chunked data shape", chunked_data.shape)
    #print(data)
    
    vq_loss, data_recon, perplexity = model(chunked_data)
    recon_error = F.mse_loss(data_recon, chunked_data) #/ data_variance
    zero_recon = F.mse_loss(torch.zeros(n//l, 1, p, l), chunked_data)
    loss = recon_error + vq_loss

    #print("recon data shape:", data_recon.shape)
    #print(data_recon)
    for i in range(data_recon.shape[0]):
        print(torch.max(data_recon[i,:,:,:]).item())
    print('Loss:', loss.item(), '\Perplexity:', perplexity.item())

    unchunked_recon = data_recon.view(p, n_2).detach().numpy()
    # Turn all negative values to 0 
    unchunked_recon = unchunked_recon.clip(min=clip_val) # min note length that should count

    if norm: # unnormalize!
        unchunked_recon = unchunked_recon * maxlength

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

if __name__ == "__main__":
    # Default paths:
    original_tensors = PROJECT_DIRECTORY / "recon_tensors"
    model_path = PROJECT_DIRECTORY / "models"
    results_folder = PROJECT_DIRECTORY / "results"
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    parser.add_argument('-t', '--tensordir', help='Path to training tensor data.', default=original_tensors)
    parser.add_argument('-m', '--modeldir', help='Path to desired model directory', default=model_path)
    parser.add_argument('-r', '--resdir', help='Path to desired result directory', default=results_folder)
    parser.add_argument('-n', '--normalize', dest='norm', action='store_const', const=True, 
                        default=False, help='whether or not to normalize the tensors')
    parser.add_argument('-b', '--batchlength', help='Length of midi object', default=256)
    args = vars(parser.parse_args())

    #loss = args['lossfunc'] # True or false
    tensor_dir = args['tensordir']
    model_name = args['modeldir']
    resdir = args['resdir']

    #fstub = args['resname']
    #issparse = args['sparse']
    normalize = args['norm']
    batchlength = int(args['batchlength'])

    print("Start")
    '''
    reconstruct_songs(tensor_dir, resdir, resdir, model_name, clip_val=0, batchlength=batchlength)
    #"Save graphs"
    save_graphs(resdir, resdir)'''

    model_path = Path('../models/new_rep/model_FINAL-2022-07-01-0.pt')

    training_set_tensors = Path('../new_recon_tensors/train_set_tensors')
    testing_set_tensors = Path('../new_recon_tensors/test_set_tensors')

    training_set_midis = Path('../new_recon_tensors/train_set_midis_wavs')
    testing_set_midis = Path('..new_recon_tensors/test_set_midis_wavs')

    training_recons = Path('../new_recon_tensors/train_recons')
    testing_recons = Path('../new_recon_tensors/test_recons')
    
    training_recon_midis = Path('../new_recon_tensors/train_recon_midis')
    testing_recon_midis = Path('../new_recon_tensors/test_recon_midis')

    print("Entering Reconstruction")
    reconstruct_songs(training_set_tensors, training_recons, training_recon_midis, model_path, clip_val=0, norm=False, batchlength=256, num_embed=1024)

    #tensors_to_midis_2(training_set_tensors, training_set_midis)
    
    #tensors_to_midis_2(testing_set_tensors, testing_set_midis)
    #print("TRAINING")
    #save_graphs(training_set_midis, training_set_midis)
    #print("TESTING")
    #save_graphs(testing_set_midis, testing_set_midis)

    # python3 listen_to_model_output.py -t "/projectnb/textconv/sadiela/midi_generation/recon_tensors/" -m "/projectnb/textconv/sadiela/midi_generation/models/new_rep/model_FINAL-2022-07-01-0.pt" -r "/projectnb/textconv/sadiela/midi_generation/models/new_rep/final_recons" -b 256
