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

import torch
import torch.nn.functional as F
from midi_utility import *
from vq_vae import * 
from vae import *
import matplotlib.pyplot as plt
import pypianoroll
import yaml
from pathlib import Path
import argparse
import pickle
from loss_functions import *

maxlength = 16*32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick device


def reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path, clip_val=0, norm=False, batchlength=256, num_embed=1024, quantize=False, embedding_dim=128):
    res_string = "MODEL FILE NAME" + str(model_path) + "\nRECON ERRORS!\n"
    file_list = os.listdir(orig_tensor_dir)

    if quantize:
        model = VQVAE_Model(num_embeddings=num_embed, embedding_dim=embedding_dim, commitment_cost=0.5)
    else: 
        model = VAE_Model(in_channels=1, hidden_dim=128*155, latent_dim=embedding_dim)

    res_string += "number of parameters in initialized model:" + str(sum(p.numel() for p in model.parameters())) + '\n'
    stat_dictionary = torch.load(model_path, map_location=torch.device(DEVICE))
    model_params = stat_dictionary["model_state_dict"]
    res_string += "number of parameters in state dictionary:" + str(sum(p.numel() for p in model_params.values())) + '\n'
    model.load_state_dict(model_params)
    model.to(DEVICE)
    model.eval()

    for file in file_list:
        print(file) # perform reconstruction
        cur_tensor, loss = reconstruct_song(Path(orig_tensor_dir) / file, model, clip_val=clip_val, norm=norm, batchlength=batchlength, quantize=quantize)
        # record info IF RECONSTRUCTION NOT ALL 0s
        if (cur_tensor > 0).sum() > 0: 
            print(cur_tensor[:,:10])
            #input("Continue")
            res_string += str(file) + ' loss: ' + str(loss.item()) # + ' zero loss:' + str(zero_loss.item()) + '\n'
            # save tensor
            sparse_arr = sparse.csr_matrix(cur_tensor) # save sparse!!!
            with open(str(Path(new_tensor_dir) / str(file.split('.')[0] + '_conv.p')), 'wb') as outfile:
                pickle.dump(sparse_arr, outfile)
            # convert to midi and save midi 
            print("entering tensor to midi")
            tensor_to_midi_2(cur_tensor, Path(new_midi_dir) / str(file.split('.')[0] + '.mid'))
        else:
            print(file, "reconstruction is all 0s")
    with open(Path(new_midi_dir) / 'recon_info.txt', 'w') as outfile:
        outfile.write(res_string)

def reconstruct_song(orig_tensor_path, model, clip_val=0.5, norm=False, batchlength=256, quantize=False):
    with open(orig_tensor_path,'rb') as f: 
        pickled_tensor = pickle.load(f)
    data = pickled_tensor.toarray()

    data = torch.tensor(data)

    # Test on a song
    #print(data.shape)
    p, n = data.shape

    l = batchlength #1024 # batch length

    data = data.to(DEVICE)

    data = data[:,:(data.shape[1]-(data.shape[1]%l))]
    p, n_2 = data.shape
    #print("Cropped data shape:", data.shape)
    data = torch.tensor(data).float()

    x = data.view((n//l, 1, p, l))
    print("chunked data shape", x.shape)
    #print(data)
    
    if quantize: 
        vq_loss, x_hat = model(x)
        recon_error = F.mse_loss(x_hat, x) #/ data_variance
        #zero_loss = F.mse_loss(torch.zeros(n//l, 1, p, l), x) + vq_loss
        loss = recon_error + vq_loss
    else: 
        x_hat, mean, log_var = model(x)
        loss = bce_loss(x_hat, x, mean, log_var)
        #zero_loss = bce_loss(torch.zeros(n//l, 1, p, l), x)

    #print("recon data shape:", data_recon.shape)
    #print(data_recon)
    for i in range(x_hat.shape[0]):
        print(torch.max(x_hat[i,:,:,:]).item())
    print('Loss:', loss.item())#, '\Perplexity:', perplexity.item())

    unchunked_recon = x_hat.view(p, n_2).detach().cpu().numpy()
    # Turn all negative values to 0 
    #unchunked_recon = unchunked_recon.clip(min=clip_val) # min note length that should count
    print(unchunked_recon)
    unchunked_recon[unchunked_recon < clip_val] = 0
    unchunked_recon[unchunked_recon >= clip_val] = 1

    print(np.sum(unchunked_recon), np.sum(data.cpu().numpy()))

    return unchunked_recon, loss #, zero_loss

def save_result_graph(yaml_file, plot_dir):
    #root_name = yaml_name.split(".")[0]
    with open(yaml_file) as file: 
        res_dic = yaml.load(file, Loader=yaml.FullLoader)
    plt.plot(res_dic['reconstruction_error'])
    plt.title("Reconstruction Error")
    plt.xlabel("Iteration")
    #plt.show()
    print("SAVING")
    plt.savefig(str(plot_dir / "recon_error.png"))

    plt.clf()

    plt.plot(res_dic['total_loss'])
    plt.title("Total Loss")
    plt.xlabel("Iteration")
    #plt.show()
    print("SAVING")
    plt.savefig(str(plot_dir / "total_loss.png"))
    plt.clf()

def save_midi_graphs(midi_path, save_path):
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

if __name__ == "__main__":
    # Default paths:
    print("GRAPHING!")
    yaml_file = '/Users/sadiela/Documents/phd/research/music/midi_generation/results/recon_error_and_perplexity_vqvae_bce_test-2022-07-09-0.yaml'
    plot_dir = Path('/Users/sadiela/Documents/phd/research/music/midi_generation/results/')
    save_result_graph(yaml_file, plot_dir)
    '''
    tensor_dir = '/projectnb/textconv/sadiela/midi_generation/new_recon_tensors/train_recons/'
    recon_res_dir = '/projectnb/textconv/sadiela/midi_generation/models/new_rep_vae_overhaul/final_recons/'
    final_model_name = '/projectnb/textconv/sadiela/midi_generation/models/new_rep_vae_overhaul/model_FINAL-2022-07-09-0.pt'
    batchlength= 256
    quantize= False
    embeddim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick device
    reconstruct_songs(str(tensor_dir), str(recon_res_dir), str(recon_res_dir), final_model_name, device=device, clip_val=0, batchlength=batchlength, quantize=quantize, embedding_dim=embeddim)
    # Save pianorolls
    save_midi_graphs(str(recon_res_dir),str(recon_res_dir))
    
    original_tensors = PROJECT_DIRECTORY / "recon_tensors"
    model_path = PROJECT_DIRECTORY / "models"
    results_folder = PROJECT_DIRECTORY / "results"
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    parser.add_argument('-t', '--tensordir', help='Path to training tensor data.', default=original_tensors)
    parser.add_argument('-m', '--modeldir', help='Path to desired model directory', default=model_path)
    parser.add_argument('-r', '--resdir', help='Path to desired result directory', default=results_folder)
    parser.add_argument('-n', '--normalize', dest='norm', action='store_const', const=True, 
                        default=False, help='whether or not to normalize the tensors')
    parser.add_argument('-c', '--reconstruct', dest='recon', action='store_const', const=True, 
                        default=False, help='whether or not to perform reconstruction')
    parser.add_argument('-s', '--savefigs', dest='save', action='store_const', const=True, 
                        default=False, help='whether or not to save pianoroll images')
    parser.add_argument('-b', '--batchlength', help='Length of midi object', default=256)
    args = vars(parser.parse_args())

    #loss = args['lossfunc'] # True or false
    tensor_dir = args['tensordir']
    model_name = args['modeldir']
    resdir = args['resdir']
    save_figs = args['save']
    reconstruct = args['recon']

    #fstub = args['resname']
    #issparse = args['sparse']
    normalize = args['norm']
    batchlength = int(args['batchlength'])

    print("Start")
    
    if reconstruct:
        reconstruct_songs(tensor_dir, resdir, resdir, model_name, clip_val=0, batchlength=batchlength)
    #"Save graphs"
    if save_figs:
        save_midi_graphs(resdir, resdir)

    #model_path = Path('../models/new_rep/model_FINAL-2022-07-01-0.pt')
    '''
    '''training_set_tensors = Path('../new_recon_tensors/train_set_tensors')
    testing_set_tensors = Path('../new_recon_tensors/test_set_tensors')

    training_set_midis = Path('../new_recon_tensors/train_set_midis_wavs')
    testing_set_midis = Path('..new_recon_tensors/test_set_midis_wavs')

    training_recons = Path('../new_recon_tensors/train_recons')
    testing_recons = Path('../new_recon_tensors/test_recons')
    
    training_recon_midis = Path('../new_recon_tensors/train_recon_midis')
    testing_recon_midis = Path('../new_recon_tensors/test_recon_midis')'''

    #reconstruct_songs(training_set_tensors, training_recons, training_recon_midis, model_path, clip_val=0, norm=False, batchlength=256, num_embed=1024)

    #tensors_to_midis_2(training_set_tensors, training_set_midis)
    
    #tensors_to_midis_2(testing_set_tensors, testing_set_midis)
    #print("TRAINING")
    #save_graphs(training_set_midis, training_set_midis)
    #print("TESTING")
    # python3 listen_to_model_output.py -t "/projectnb/textconv/sadiela/midi_generation/recon_tensors/" -m "/projectnb/textconv/sadiela/midi_generation/models/new_rep/model_FINAL-2022-07-01-0.pt" -r "/projectnb/textconv/sadiela/midi_generation/models/new_rep/final_recons" -b 256
