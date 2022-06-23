from midi_utility import PROJECT_DIRECTORY
from vq_vae import *
from gen_utility import * 
import torch
import yaml
import argparse
import time
from pathlib import Path
import logging

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
datpath = PROJECT_DIRECTORY  / 'data' / 'all_midi_tensors_ttv' / 'train'  # '..\\midi_data\\full_dataset_midis_normalized\\'
desktopdatpath = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'midi_tensors_2'
modpath = PROJECT_DIRECTORY / 'models'
respath = PROJECT_DIRECTORY / 'results'
logpath = PROJECT_DIRECTORY / 'scripts' / 'log_files'
#/usr3/graduate/sadiela/midi_generation/models/' #model_10_25_2.pt'


##### HYPERPARAMETERS TO SPECIFY #####
# loss = str, which loss function to use (mse or l1reg)
# datadir = str, directory where training data is stored
# modeldir = str, directory to save trained model
# fstub = str, stub for output files
# outdir = str, where to put results (loss & perplex throughout training)
# sparse = (t/f) whether data is stored sparsely
# batchsize = int, Number of songs in a given batch
# batchlength = int, Length of a midi object
# normalize = (t/f) whether to 0-1 normalize MIDI data
# quantize = (t/f) VAE vs VQ-VAE
# embeddim = int, dimension of embeddings
# numembed = int, total number of embeddings
# lr = 1e-3 learning rate

##############################
# MODEL/OPTIMIZER PARAMETERS #
##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(datapath, resultspath, modelpath, fstub, loss, lr=1e-3, batchsize=10, batchlength=256, normalize=False, quantize=True, sparse=False, num_embeddings=1024, embedding_dim=128, lam=1):
    print("DEVICE:", device)
    ### Declare model ###
    #model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5, quantize=quantize).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    logging.info("Models will be saved in the directory: %s", modelpath)

    # train model
    recon_error, perplex, nan_recon_files = train_model(datapath, modelpath, 
                                                        num_embeddings=num_embeddings, 
                                                        embedding_dim=embedding_dim, 
                                                        learning_rate=lr, lossfunc=loss, 
                                                        bs=batchsize, batchlength=batchlength, 
                                                        normalize=normalize, quantize=quantize, 
                                                        sparse=sparse, lam=lam)
    
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
    parser.add_argument('-r', '--resname', help='Result and model stub', default="VQ_VAE_training")
    parser.add_argument('-n', '--normalize', dest='norm', action='store_const', const=True, 
                        default=False, help='whether or not to normalize the tensors')
    parser.add_argument('-b', '--batchsize', help='Number of songs in a batch', default=5)
    parser.add_argument('-a', '--batchlength', help='Length of midi object', default=256)
    parser.add_argument('-u', '--numembed', help='Number of embeddings', default=1024)
    parser.add_argument('-e', '--embeddim', help='Embedding dimension', default=128)

    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('-l','--lossfunc', help='loss function to use', default='mse')
    parser.add_argument('-q', '--quantize', dest='quant', action='store_const', const=False,
                        default=True, help="True=VQVAE, false=VAE")
    #parser.add_argument('-n', '--nfolds', help='number of folds to use in cross validation', default=1) # make default 1?
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='INFO',help='specify level of detail for log file')
    parser.add_argument('-s', '--sparse', dest='sparse', action='store_const', const=True, 
                        default=False, help='whether or not to tensors are sparse')
    parser.add_argument('-k', '--lambda', help='L1 Hyperparam', default='1')
    args = vars(parser.parse_args())

    loglevel = args['loglevel']
    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase

    logfile = get_free_filename('vq_vae_training_log', logpath, suffix='.log')

    logging.basicConfig(filename=logfile, level=numeric_level)

    logging.info("ARGS: %s", str(args))
    #input("Continue...")
    loss = args['lossfunc'] # True or false
    datadir = args['datadir']
    modeldir = args['modeldir']
    outdir = args['outdir']
    fstub = args['resname']
    sparse = args['sparse']
    normalize = args['norm']
    quantize = args['quant']
    batchsize = int(args['batchsize'])
    batchlength = int(args['batchlength'])
    embeddim = int(args['embeddim'])
    numembed = int(args['numembed'])
    lam = int(args["lambda"])
    lr = 1e-3

    logging.info("Chosen hyperparameters:")
    logging.info("Loss function: %s", loss)
    logging.info("Data directory:%s \nModel directory:%s \nOutput directory:%s", datadir, modeldir, outdir)
    logging.info("File stub:%s",fstub)
    logging.info("Sparse:%s\nNormalize:%s\nQuantize:%s\n", str(sparse), str(normalize), str(quantize))
    logging.info("Batch size:%s\nBatch length:%s\n", batchsize, batchlength)
    logging.info("Embedding dimension:%s\nNumber of embeddings:%s\n", embeddim, numembed)
    logging.info("Learning Rate:%s", lr)
    logging.info("Lambda:%d", lam)

    print("Chosen hyperparameters:")
    print("Loss function:", loss)
    print("Data directory:",datadir, "\nModel directory:",modeldir, "\nOutput directory:", outdir)
    print("File stub:",fstub)
    print("Sparse:", sparse, "\nNormalize:",normalize,"\nQuantize:",quantize)
    print("Batch size:%s\nBatch length:%s\n", batchsize, batchlength)
    print("Embedding dimension:", embeddim,"\nNumber of embeddings:", numembed)
    print("Lambda:", lam)
    print("Learning Rate:", lr)
    
    train(datadir, outdir, modeldir, fstub, loss, lr=lr, batchsize=batchsize, batchlength=batchlength, normalize=normalize, quantize=quantize, sparse=sparse, num_embeddings=numembed, embedding_dim=embeddim, lam=lam)
    logging.info("All done! TOTAL TIME: %s", str(time.time()-prog_start))
