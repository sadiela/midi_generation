from midi_utility import PROJECT_DIRECTORY
from vq_vae import *
from vae import *
from gen_utility import * 
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import time
from pathlib import Path
import logging
from reconstruct_from_model import *


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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick device
PITCH_DIM = 128

#####################
# DEFAULT FILEPATHS #
#####################
#data_folder = Path("source_data/text_files/")
datadir = PROJECT_DIRECTORY  / 'data' / 'ttv'  # '..\\midi_data\\full_dataset_midis_normalized\\'
desktopdatdir = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'midi_tensors_2'
modelupperdir = PROJECT_DIRECTORY / 'models'
resdir = PROJECT_DIRECTORY / 'models'
#logpath = PROJECT_DIRECTORY / 'scripts' / 'log_files'
#/usr3/graduate/sadiela/midi_generation/models/' #model_10_25_2.pt'

###############################
# CUSTOM DATALOADER/COLLATION #
###############################
class MidiDataset(Dataset):
    """Midi dataset."""

    def __init__(self, npy_file_dir, l=256):
        """
        Args:
            npy_file_dir (string): Path to the npy file directory
        """
        file_list = os.listdir(npy_file_dir)
        self.l = l
        self.maxlength = 16*64
        self.paths = [ Path(npy_file_dir) / file for file in file_list] # get entire list of midi tensor file names 
        
    def __getitem__(self, index):
        # choose random file path from directory (not already chosen), chunk it 
        #cur_data = torch.load(self.paths[index])
        #print(self.paths[index])
        with open(self.paths[index], 'rb') as f:
          pickled_tensor = pickle.load(f)
        cur_data = torch.tensor(pickled_tensor.toarray()).float()

        p, l_i = cur_data.shape
        
        # make sure divisible by l
        # CHUNK! 
        #print("DATA SHAPE:", cur_data.shape)
        if l_i // self.l == 0: 
          padded_data = torch.zeros((p, self.l))
          padded_data[:,0:l_i] = cur_data
          l_i=self.l
        else: 
          padded_data = cur_data[:,:(cur_data.shape[1]-(cur_data.shape[1]%self.l))]
        
        chunked = torch.reshape(padded_data, (l_i//self.l,1, p, self.l)) 
        # Remove empty areas
        chunked = chunked[chunked.sum(dim=(2,3)) != 0]
        chunked = torch.reshape(chunked, (chunked.shape[0], 1, p, self.l)) 

        if chunked.shape[0] != 0:
            return chunked # 3d tensor: l_i\\l x p x l
        else:
            return None
        return padded

    def __getname__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

def collate_fn(data, collate_shuffle=True):
  # data is a list of tensors; concatenate and shuffle all list items
  #print(len(data), type(data))
  #print(data)
  data = list(filter(lambda x: x is not None, data))
  #print(len(data))
  full_list = torch.cat(data, 0) # concatenate all data along the first axis
  if collate_shuffle:
    idx = torch.randperm(full_list.shape[0])
    return  full_list[idx].view(full_list.size())
  else:
    return full_list


def train_model(datapath, model_save_path, num_embeddings=1024, embedding_dim=128, learning_rate=1e-3, lossfunc='mse', batchsize=10, batchlength=256, normalize=False, quantize=True, lam=1, epochs=1):
    print("DEVICE:", DEVICE)
    ### Declare model ###
    #model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5, quantize=quantize).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    logging.info("Models will be saved in the directory: %s", model_save_path)

    midi_tensor_dataset = MidiDataset(Path(datapath) / "train", l=batchlength) # dataset declaration

    ### Declare model ### 
    if quantize:
        model = VQVAE_Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5)
    else: 
        model = VAE_Model(in_channels=1, hidden_dim=PITCH_DIM*155, latent_dim=embedding_dim)

    ### Declare optimizer ###
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False) # optimizer declaration

    model.to(DEVICE)
    model.float()
    model.train() # training mode
    train_res_recon_error = []
    train_res_perplexity = []
    train_res_total_loss = []

    training_data = DataLoader(midi_tensor_dataset, collate_fn=collate_fn, batch_size=batchsize, shuffle=True, num_workers=2)
    # Let # of tensors = n
    # each tensor is pxl_i, where l_i is the length of the nth tensor
    # when we chunk the data, it becomes (l_i//l = s_i) x 1 x p x l 
    # so we want a big (sum(s_i)) x 1 x p x l tensor. 

    max_tensor_size= 0 
    #dynamic_loss = SpeedySparseDynamicLoss.apply
    model_number = 1
    for e in range(epochs):
      # train loop
      for i, x in tqdm(enumerate(training_data)):
          #print(i)
          #name = midi_tensor_xset.__getname__(i)
          #optimizer.zero_grad() # yes? 
          # s x p x 1 x l
          x = x.to(DEVICE)
          cursize = torch.numel(x)
          if cursize > max_tensor_size:
            max_tensor_size = cursize
            logging.info("NEW MAX BATCH SIZE: %d", max_tensor_size)

                  #print('TRAIN:', x.shape)
          if quantize: 
            x_hat, vq_loss, perplexity = model(x)
            recon_error = calculate_recon_error(x_hat, x, lossfunc=lossfunc, lam=lam)
            loss = recon_error + vq_loss # will be 0 if no quantization
          else:
            x_hat, mean, log_var = model(x)
            recon_error = calculate_recon_error(x_hat, x, lossfunc=lossfunc, lam=lam) 
            loss = recon_error +  kld(mean, log_var)

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
          optimizer.step()
          
          ### RECORD LOSSES ###
          train_res_total_loss.append(loss.item())
          train_res_recon_error.append(recon_error.item())


          if (i) % 5000 == 0:
            # new save path
            cur_model_file = get_free_filename('model_'+str(model_number), model_save_path, suffix='.pt')
            model_number+=1
            torch.save({
                        'epoch': e,
                        'iteration': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'recon_loss': np.mean(train_res_recon_error[-5000:]),
                        'total_loss': np.mean(train_res_total_loss[-5000:])
                        }, cur_model_file) # incremental saves
            logging.info('%d iterations' % (i+1))
            logging.info('recon_error: %.3f' % np.mean(train_res_recon_error[-5000:]))
            logging.info('\n')
            print(i, 'iterations')
            print('recon_error:', np.mean(train_res_recon_error[-5000:]))

      # VALIDATION LOOP

    final_model_file = get_free_filename('model_FINAL', model_save_path, suffix='.pt')
    logging.info("saving model to %s"%final_model_file)
    torch.save(model.state_dict(), final_model_file)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(train_res_recon_error[-5000:]),
                }, final_model_file) 
    return train_res_recon_error, train_res_total_loss, train_res_perplexity, final_model_file


if __name__ == "__main__":
    prog_start = time.time()
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    #parser.add_argument('-l','--lossfunc', help='Choose loss function.', default=True)
    parser.add_argument('-d', '--datadir', help='Path to training tensor data.', default=datadir)
    parser.add_argument('-m', '--modeldir', help='desired model subdirectory name', required=True) # default=modeldir)
    parser.add_argument('-r', '--resname', help='Result and model stub', default="VQ_VAE_training")
    parser.add_argument('-b', '--batchsize', help='Number of songs in a batch', default=5)
    parser.add_argument('-a', '--batchlength', help='Length of midi object', default=256) # want to change to 384 eventually
    parser.add_argument('-u', '--numembed', help='Number of embeddings', default=1024)
    parser.add_argument('-e', '--embeddim', help='Embedding dimension', default=128)
    parser.add_argument('-t', '--epochs', help='Number of training epochs', default=1)

    # run a single label experiment by default, if --multi flag is added, run a multilabel experiment!
    parser.add_argument('-l','--lossfunc', help='loss function to use', default='mse')
    parser.add_argument('-q', '--quantize', dest='quant', action='store_const', const=True,
                        default=False, help="True=VQVAE, false=VAE")
    #parser.add_argument('-n', '--nfolds', help='number of folds to use in cross validation', default=1) # make default 1?
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='INFO',help='specify level of detail for log file')
    parser.add_argument('-k', '--lambda', help='L1 Hyperparam', default='1')
    args = vars(parser.parse_args())

    loglevel = args['loglevel']
    
    #input("Continue...")
    loss = args['lossfunc'] # True or false
    datadir = args['datadir']
    modelsubdir = args['modeldir']
    fstub = args['resname']
    quantize = args['quant']
    batchsize = int(args['batchsize'])
    batchlength = int(args['batchlength'])
    embeddim = int(args['embeddim'])
    numembed = int(args['numembed'])
    lam = float(args["lambda"])
    epochs = int(args['epochs'])
    lr = 1e-3

    # create directory for models and results
    modeldir = modelupperdir / modelsubdir
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)

    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase
    logfile = get_free_filename('vq_vae_training_log', modeldir, suffix='.log')

    logging.basicConfig(filename=logfile, level=numeric_level)

    hyperparameters = '\nData directory' + str(datadir) + '\nModel/output directory' + str(modeldir) + '\nFile stub:' + fstub
    hyperparameters += '\nQuantize:' + str(quantize)
    hyperparameters += '\nBatch size:' + str(batchsize) + '\nBatch length:' + str(batchlength)
    hyperparameters +=  '\nEmbedding dimension:' + str(embeddim) + '\nNumber of embeddings:' + str(numembed)
    hyperparameters += '\nLearning rate:' + str(lr) + '\nLambda:' + str(lam)

    logging.info("Chosen hyperparameters:")
    logging.info(hyperparameters)

    print("Chosen hyperparameters:")
    print(hyperparameters)

    model_hyperparameters = {
        "datadir": str(datadir), 
        "modeldir": str(modeldir),
        "fstub": fstub,
        "quantize": quantize,
        "batchsize": batchsize,
        "batchlength": batchlength,
        "embeddingdim": embeddim,
        "numembed": numembed,
        "lossfunc": loss,
        "learningrate": lr, 
        "lambda": lam
    }

    # save hyperparameters to YAML file in folder
    param_file = modeldir / "MODEL_PARAMS.yaml"
    try:
        with open(str(param_file), 'w') as outfile: 
            yaml.dump(model_hyperparameters, outfile)
    except Exception as e: 
        print(e)

    #####################
    #### Train model ####
    #####################

    recon_error, total_loss, perplex, final_model_name = train_model(datadir, modeldir, num_embeddings=numembed, embedding_dim=embeddim, lossfunc=loss, learning_rate=lr, batchsize=batchsize, batchlength=batchlength, quantize=quantize, lam=lam, epochs=epochs)
    logging.info("All done training! TOTAL TIME: %s", str(time.time()-prog_start))

    # save losses to file and plot graph!
    results={"reconstruction_error": recon_error, "total_loss": total_loss, "perplexity": perplex}
    savefile = get_free_filename('recon_error_and_perplexity_' + fstub, modeldir, suffix='.yaml')
    logging.info("SAVING FILE TO: %s", savefile)
    try: 
        with open(savefile, 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)
        save_result_graph(fstub, savefile, modeldir)
    except Exception as e: 
        print(e)

    ########################
    # Reconstruct tensors! #
    ######################## 
    recontensordir = Path('../new_recon_tensors/train_set_tensors/')
    reconresdir = Path(modeldir) / 'final_recons'
    if not os.path.isdir(reconresdir):
        os.mkdir(reconresdir)

    # reconstruct midis
    reconstruct_songs(model_hyperparameters, reconresdir, recontensordir, final_model_name, clip_val=0.5)
    # Save pianorolls
    save_midi_graphs(str(reconresdir),str(reconresdir))

    # python3 train_vqvae.py -d '../mini_data' -m 'mini_vq_mse' -r "mini_vq_mse" -l "mse" -q
    # python3 train_vqvae.py -d '../mini_data' -m 'mini_vq_bce' -r "mini_vq_bce" -l "bce" -q
    # python3 train_vqvae.py -d '../mini_data' -m 'mini_vae_bce' -r "mini_vae_bce" -l "bce"
    # python3 train_vqvae.py -d '../mini_data' -m 'mini_vq_l1reg' -r "mini_vq_l1reg" -l "l1reg" -t 20 -q
