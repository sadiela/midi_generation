from midi_utility import PROJECT_DIRECTORY
from vq_vae import *
from gen_utility import * 
import torch
import yaml
import argparse
import time
from pathlib import Path

#############
# FILEPATHS #
#############
#data_folder = Path("source_data/text_files/")
datpath = PROJECT_DIRECTORY / 'data' / 'all_midi_tensors'  # '..\\midi_data\\full_dataset_midis_normalized\\'
modpath = PROJECT_DIRECTORY / 'models'
respath = PROJECT_DIRECTORY / 'results'
#/usr3/graduate/sadiela/midi_generation/models/' #model_10_25_2.pt'

##############################
# MODEL/OPTIMIZER PARAMETERS #
##############################
batch_size = 256
num_training_updates = 15000
num_hiddens = 128
num_residual_hiddens = 16
num_residual_layers = 2
l = 1024 # batch length
decay = 0.99
learning_rate = 1e-3
num_embeddings = 64
embedding_dim = 32
commitment_cost = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(datapath, resultspath, modelpath, fstub, mse_loss):
    # i think num embeddings was 64 before? 
    model = Model(num_embeddings=1024, embedding_dim=128, commitment_cost=0.5).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    model_file = get_free_filename('model_' + fstub, modelpath, suffix='.pt')
    recon_error, perplex, nan_recon_files = train_model(datapath, model, model_file, mse_loss=mse_loss)
    # save losses to file
    print("NUM NAN FILES:", len(nan_recon_files))
    results={"reconstruction_error": recon_error, "perplexity": perplex, "nan_reconstruction_files": nan_recon_files}
    savefile = get_free_filename('results_' + fstub, resultspath, suffix='.yaml')
    print("SAVING FILE TO:", savefile)
    with open(savefile, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

if __name__ == "__main__":
    prog_start = time.time()
    print("START")
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    #parser.add_argument('-l','--lossfunc', help='Choose loss function.', default=True)
    parser.add_argument('-d', '--datadir', help='Path to training tensor data.', default=datpath)
    parser.add_argument('-m', '--modeldir', help='Path to desired model directory', default=modpath)
    parser.add_argument('-o', '--outdir', help='Path to desired model directory', default=respath)
    parser.add_argument('-n', '--resname', help='Result and model stub', default="")
    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('-l','--lossfunc', dest='lossfunction', action='store_const', const=False,
                        default=True, help="True=mse, false=l1")
    #parser.add_argument('-n', '--nfolds', help='number of folds to use in cross validation', default=1) # make default 1?
    #parser.add_argument('-f', '--fullres', help='generate full result file.', dest='result',
    #                    action='store_const', const='full', default='summary')
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='WARNING',help='specify level of detail for log file')
    # IMPLEMENT THIS!
    #parser.add_argument('-l' '--labels', dest='print_labels', action='store_const', const=True, default=False,
    #                    help='Print missed labels')

    args = vars(parser.parse_args())
    print("ARGS:", args)
    #input("Continue...")
    mse_loss = args['lossfunction'] # True or false
    datadir = args['datadir']
    modeldir = args['modeldir']
    fstub = args['resname']
    outdir = args['outdir']
    
    test(datadir, outdir, modeldir, fstub, mse_loss)
    print("All done!")
    ### NEED TO TEST PATHLIB !!!