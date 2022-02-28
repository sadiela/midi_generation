from midi_utility import PROJECT_DIRECTORY
from vq_vae import *
from gen_utility import * 
import torch
import yaml
import argparse
import time
from pathlib import Path
import logging
from datetime import datetime


'''
Driver script for training VQ-VAE models. Takes the following command line arguments (all optional):
-d: path to desired training data directory
-m: path to directory where you want to save the model
-o: path to directory where you want to save results (loss/complexity over training period)
-r: filename stub for model and result files
-n: (binary) whether or not to normalize the midi tensors
-b: batchsize, i.e. the number of songs in a batch
-l: (binary) desired loss function (no flag = mse, flag = mae)
-q: (binary) whether or not to quantize the representation z
-v: verbosity; indicates level of detail for logging information 
-s: whether or note the training data is stored as sparse matrices
'''

#####################
# DEFAULT FILEPATHS #
#####################
#data_folder = Path("source_data/text_files/")
datpath = PROJECT_DIRECTORY  / 'data' / 'all_midi_tensors'  # '..\\midi_data\\full_dataset_midis_normalized\\'
desktopdatpath = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'midi_tensors_2'
modpath = PROJECT_DIRECTORY / 'models'
respath = PROJECT_DIRECTORY / 'results'
logpath = PROJECT_DIRECTORY / 'scripts' / 'log_files'
#/usr3/graduate/sadiela/midi_generation/models/' #model_10_25_2.pt'

##############################
# MODEL/OPTIMIZER PARAMETERS #
##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(datapath, resultspath, modelpath, fstub, loss, batchsize=10, batchlength=256, normalize=False, quantize=True, sparse=False, num_embeddings=1024, embedding_dim=36):
    # i think num embeddings was 64 before? 
    # Declare model
    print("DEVICE:", device)
    model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5, quantize=quantize).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    model_file = get_free_filename('model_' + fstub, modelpath, suffix='.pt')
    logging.info("Model will be saved to: %s", model_file)

    # train model
    recon_error, perplex, nan_recon_files = train_model(datapath, model, model_file, lossfunc=loss, bs=batchsize, batchlength=batchlength, normalize=normalize, quantize=quantize, sparse=sparse)
    
    # save losses to file
    logging.info("NUM NAN FILES: %d", len(nan_recon_files))
    results={"reconstruction_error": recon_error, "perplexity": perplex, "nan_reconstruction_files": nan_recon_files}
    savefile = get_free_filename('results_' + fstub, resultspath, suffix='.yaml')
    logging.info("SAVING FILE TO: %s", savefile)
    with open(savefile, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

if __name__ == "__main__":
    prog_start = time.time()
    print("START")
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    #parser.add_argument('-l','--lossfunc', help='Choose loss function.', default=True)
    parser.add_argument('-d', '--datadir', help='Path to training tensor data.', default=datpath)
    parser.add_argument('-m', '--modeldir', help='Path to desired model directory', default=modpath)
    parser.add_argument('-o', '--outdir', help='Path to desired result directory', default=respath)
    parser.add_argument('-r', '--resname', help='Result and model stub', default="")
    parser.add_argument('-n', '--normalize', dest='norm', action='store_const', const=True, 
                        default=False, help='whether or not to normalize the tensors')
    parser.add_argument('-b', '--batchsize', help='Number of songs in a batch', default=10)
    parser.add_argument('-a', '--batchlength', help='Length of midi object', default=128)
    parser.add_argument('-u', '--numembed', help='Number of embeddings', default=1024)
    parser.add_argument('-e', '--embeddim', help='Number of embeddings', default=36)

    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('-l','--lossfunc', help='loss function to use', default='mse')
    parser.add_argument('-q', '--quantize', dest='quant', action='store_const', const=False,
                        default=True, help="True=VQVAE, false=VAE")
    #parser.add_argument('-n', '--nfolds', help='number of folds to use in cross validation', default=1) # make default 1?
    #parser.add_argument('-f', '--fullres', help='generate full result file.', dest='result',
    #                    action='store_const', const='full', default='summary')
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='INFO',help='specify level of detail for log file')
    # IMPLEMENT THIS!
    #parser.add_argument('-l' '--labels', dest='print_labels', action='store_const', const=True, default=False,
    #                    help='Print missed labels')
    parser.add_argument('-s', '--sparse', dest='sparse', action='store_const', const=True, 
                        default=False, help='whether or not to tensors are sparse')

    args = vars(parser.parse_args())

    loglevel = args['loglevel']
    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase

    now = datetime.now()
    date = now.strftime("%m-%d-%Y")
    logfile = get_free_filename('vq_vae_training-' + date, logpath, suffix='.log')

    logging.basicConfig(filename=logfile, encoding='utf-8', level=numeric_level)

    logging.info("ARGS: %s", str(args))
    #input("Continue...")
    loss = args['lossfunc'] # True or false
    datadir = args['datadir']
    modeldir = args['modeldir']
    fstub = args['resname']
    outdir = args['outdir']
    sparse = args['sparse']
    batchsize = int(args['batchsize'])
    batchlength = int(args['batchlength'])
    normalize = args['norm']
    quantize = args['quant']
    embeddim = int(args['embeddim'])
    numembed = int(args['numembed'])
    
    train(datadir, outdir, modeldir, fstub, loss, batchsize=batchsize, batchlength=batchlength, normalize=normalize, quantize=quantize, sparse=sparse, num_embeddings=numembed, embedding_dim=embeddim)
    logging.info("All done! TOTAL TIME: %s", str(time.time()-prog_start))
