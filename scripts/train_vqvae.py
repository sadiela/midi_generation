from midi_utility import PROJECT_DIRECTORY
from vq_vae import *
from gen_utility import * 
import torch
import yaml
import argparse
import time
from pathlib import Path
import logging
from listen_to_model_output import *

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
respath = PROJECT_DIRECTORY / 'models'
#logpath = PROJECT_DIRECTORY / 'scripts' / 'log_files'
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

def train(datapath, modelpath, fstub, loss, lr=1e-3, batchsize=10, batchlength=256, normalize=False, quantize=True, sparse=False, num_embeddings=1024, embedding_dim=128, lam=1):
    print("DEVICE:", device)
    ### Declare model ###
    #model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5, quantize=quantize).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    logging.info("Models will be saved in the directory: %s", modelpath)

    # train model
    recon_error, perplex, final_model_name = train_model(datapath, modelpath, 
                                                        num_embeddings=num_embeddings, 
                                                        embedding_dim=embedding_dim, 
                                                        learning_rate=lr, lossfunc=loss, 
                                                        bs=batchsize, batchlength=batchlength, 
                                                        normalize=normalize, quantize=quantize, 
                                                        sparse=sparse, lam=lam)
    
    # save losses to file
    results={"reconstruction_error": recon_error, "perplexity": perplex}
    savefile = get_free_filename('results_' + fstub, modelpath, suffix='.yaml')
    logging.info("SAVING FILE TO: %s", savefile)
    with open(savefile, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)
    save_result_graph(savefile, model_path)
    return final_model_name, savefile

if __name__ == "__main__":
    prog_start = time.time()
    print("START")
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    #parser.add_argument('-l','--lossfunc', help='Choose loss function.', default=True)
    parser.add_argument('-d', '--datadir', help='Path to training tensor data.', default=datpath)
    parser.add_argument('-m', '--modeldir', help='Path to desired model directory', default=modpath)
    #parser.add_argument('-o', '--outdir', help='Path to desired result directory', default=respath)
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
    
    #input("Continue...")
    loss = args['lossfunc'] # True or false
    datadir = args['datadir']
    modeldir = args['modeldir']
    #outdir = args['outdir']
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

    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase

    logfile = get_free_filename('vq_vae_training_log', modeldir, suffix='.log')

    logging.basicConfig(filename=logfile, level=numeric_level)

    hyperparameters = '\nData directory' + datadir + '\nModel/output directory' + modeldir + '\nFile stub:' + fstub
    hyperparameters += '\nSparse:' + str(sparse) + "\nNormalize:" + str(normalize) + '\nQuantize:' + str(quantize)
    hyperparameters += '\nBatch size:' + str(batchsize) + '\nBatch length:' + str(batchlength)
    hyperparameters +=  '\nEmbedding dimension:' + str(embeddim) + '\nNumber of embeddings:' + str(numembed)
    hyperparameters += '\nLearning rate:' + str(lr) + '\nLambda:' + str(lam)

    logging.info("Chosen hyperparameters:")
    logging.info(hyperparameters)

    print("Chosen hyperparameters:")
    print(hyperparameters)
    
    final_model_name, yaml_name = train(datadir, modeldir, fstub, loss, lr=lr, batchsize=batchsize, batchlength=batchlength, normalize=normalize, quantize=quantize, sparse=sparse, num_embeddings=numembed, embedding_dim=embeddim, lam=lam)
    logging.info("All done! TOTAL TIME: %s", str(time.time()-prog_start))


    # path 
    tensor_dir = Path('..') / 'new_recon_tensors'/ 'train_set_tensors'
    recon_res_dir = Path(modeldir) / 'final_recons'
    try: 
        recon_res_dir.mkdir()
    except OSError as error: 
        print(error)  

    reconstruct_songs(str(tensor_dir), str(recon_res_dir), str(recon_res_dir), final_model_name, clip_val=0, batchlength=batchlength)
    #"Save graphs"
    save_graphs(str(recon_res_dir),str(recon_res_dir))

    show_result_graphs(modeldir, yaml_name, modeldir)
