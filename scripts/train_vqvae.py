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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pick device
PITCH_DIM = 128

#####################
# DEFAULT FILEPATHS #
#####################
#data_folder = Path("source_data/text_files/")
datpath = PROJECT_DIRECTORY  / 'data' / 'ttv'  # '..\\midi_data\\full_dataset_midis_normalized\\'
desktopdatpath = PROJECT_DIRECTORY / 'midi_data' / 'new_data' / 'midi_tensors_2'
modpath = PROJECT_DIRECTORY / 'models'
respath = PROJECT_DIRECTORY / 'models'
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
        #print(str(self.paths[index]))
        # load in tensor
        with open(self.paths[index], 'rb') as f:
          pickled_tensor = pickle.load(f)
        cur_tensor = pickled_tensor.toarray()
        cur_data = torch.tensor(cur_tensor)
        p, l_i = cur_data.shape
        
        # make sure divisible by l
        # CHUNK! 
        #print("DATA SHAPE:", cur_data.shape)
        if l_i // self.l == 0: 
          padded = torch.zeros((p, self.l))
          padded[:,0:l_i] = cur_data
          l_i=self.l
        else: 
          padded = cur_data[:,:(cur_data.shape[1]-(cur_data.shape[1]%self.l))]
        padded = padded.float()
        cur_chunked = torch.reshape(padded, (l_i//self.l, 1, p, self.l)) 
        
        return cur_chunked # 3d tensor: l_i\\l x p x l

    def __getname__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

def collate_fn(data, collate_shuffle=True):
  # data is a list of tensors; concatenate and shuffle all list items
  full_list = torch.cat(data, 0)
  if collate_shuffle:
    idx = torch.randperm(full_list.shape[0])
    return  full_list[idx].view(full_list.size())
  else:
    return full_list

def bce_loss(x_hat, x, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    kld = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + kld

def validate_model(model, data_path, batchlength= 256, batchsize=5, num_embeddings=1024, lossfunc='mse', lam=1):
    midi_tensor_validation = MidiDataset(data_path, l=batchlength) 
    validation_data = DataLoader(midi_tensor_validation, collate_fn=collate_fn, batch_size=batchsize, shuffle=True, num_workers=2)

    model.eval()

    validation_recon_error = []

    for i, v_x in enumerate(validation_data):
      v_x = v_x.to(DEVICE)
      vq_loss, x_hat, perplexity = model(v_x)
      if lossfunc=='mse':
        recon_error = F.mse_loss(x_hat, v_x)
      elif lossfunc=='l1reg':
        recon_error = F.mse_loss(x_hat, v_x) + (1.0/v_x.shape[0])*lam*torch.norm(x_hat, p=1) # +  ADD L1 norm
      else: # loss function = mae
        recon_error = F.l1_loss(x_hat, v_x)
      validation_recon_error.append(recon_error)
      if (i+1) % 10 == 0:
        print("Val recon:", recon_error)
        #logging.info('validation recon_error: %.3f' % np.mean(validation_recon_error[-1]))

    model.train()


def train_model(datapath, model_save_path, num_embeddings=1024, embedding_dim=128, learning_rate=1e-3, lossfunc='mse', bs=10, batchlength=256, normalize=False, quantize=True, lam=1):
    print("DEVICE:", DEVICE)
    ### Declare model ###
    #model = Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5, quantize=quantize).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    logging.info("Models will be saved in the directory: %s", model_save_path)

    midi_tensor_dataset = MidiDataset(Path(datapath) / "train", l=batchlength, norm=normalize) # dataset declaration

    ### Declare model ### 
    if quantize:
        model = VQVAE_Model(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.5, quantize=quantize).to(DEVICE) 
    else: 
        model = VAE_Model(in_channels=1, hidden_dim=PITCH_DIM*155, latent_dim=embedding_dim).to(DEVICE)

    ### Declare optimizer ###
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False) # optimizer declaration

    model.float()
    model.train() # training mode
    train_res_recon_error = []
    train_res_perplexity = []
    total_loss = []

    training_data = DataLoader(midi_tensor_dataset, collate_fn=collate_fn, batch_size=bs, shuffle=True, num_workers=2)
    # Let # of tensors = n
    # each tensor is pxl_i, where l_i is the length of the nth tensor
    # when we chunk the data, it becomes (l_i//l = s_i) x 1 x p x l 
    # so we want a big (sum(s_i)) x 1 x p x l tensor. 

    max_tensor_size= 0 
    #dynamic_loss = SpeedySparseDynamicLoss.apply
    model_number = 1
    for e in range(2):
      # train loop
      for i, x in tqdm(enumerate(training_data)):
          #name = midi_tensor_xset.__getname__(i)
          optimizer.zero_grad() # yes? 
          # s x p x 1 x l
          x = x.to(DEVICE)
          cursize = torch.numel(x)
          if cursize > max_tensor_size:
            max_tensor_size = cursize
            logging.info("NEW MAX BATCH SIZE: %d", max_tensor_size)

          #print('TRAIN:', x.shape)
          if quantize: 
            vq_loss, x_hat, perplexity = model(x)
            if lossfunc=='mse':
                recon_error = F.mse_loss(x_hat, x) #/ data_variance
            #elif lossfunc=='dyn':
            #  recon_error = dynamic_loss(x_hat, x, device) #X_hat, then X!!!
            elif lossfunc=='l1reg':
                recon_error = F.mse_loss(x_hat, x) + (1.0/x.shape[0])*lam*torch.norm(x_hat, p=1) # +  ADD L1 norm
            elif lossfunc=='bce':
                recon_error = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
            else: # loss function = mae
                recon_error = F.l1_loss(x_hat, x)
            loss = recon_error + vq_loss # will be 0 if no quantization
          else:
            x_hat, mean, log_var = model(x)
            loss = bce_loss(x_hat, x, mean, log_var)


          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
          optimizer.step()
          
          ### RECORD LOSSES ###
          total_loss.append(loss.item())
          if quantize:
            train_res_recon_error.append(recon_error.item())
            #train_res_perplexity.append(perplexity.item())
          else:
            train_res_recon_error.append(loss.item())
            #train_res_perplexity.append(0)

          if (i) % 5000 == 0:
            # new save path
            cur_model_file = get_free_filename('model_'+str(model_number), model_save_path, suffix='.pt')
            model_number+=1
            torch.save({
                        'epoch': e,
                        'iteration': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.mean(train_res_recon_error[-5000:]),
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
    return train_res_recon_error, train_res_perplexity, final_model_file


if __name__ == "__main__":
    prog_start = time.time()
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    #parser.add_argument('-l','--lossfunc', help='Choose loss function.', default=True)
    parser.add_argument('-d', '--datadir', help='Path to training tensor data.', default=datpath)
    parser.add_argument('-m', '--modeldir', help='desired model subdirectory name', default=modpath)
    parser.add_argument('-r', '--resname', help='Result and model stub', default="VQ_VAE_training")
    parser.add_argument('-b', '--batchsize', help='Number of songs in a batch', default=5)
    parser.add_argument('-a', '--batchlength', help='Length of midi object', default=256) # want to change to 384 eventually
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
    modelsubdir = args['modeldir']
    fstub = args['resname']
    sparse = args['sparse']
    quantize = args['quant']
    batchsize = int(args['batchsize'])
    batchlength = int(args['batchlength'])
    embeddim = int(args['embeddim'])
    numembed = int(args['numembed'])
    lam = int(args["lambda"])
    lr = 1e-3

    # create directory for models and results
    modeldir = modpath / modelsubdir
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)

    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase
    logfile = get_free_filename('vq_vae_training_log', modeldir, suffix='.log')

    logging.basicConfig(filename=logfile, level=numeric_level)

    hyperparameters = '\nData directory' + datadir + '\nModel/output directory' + modeldir + '\nFile stub:' + fstub
    hyperparameters += '\nSparse:' + str(sparse) + '\nQuantize:' + str(quantize)
    hyperparameters += '\nBatch size:' + str(batchsize) + '\nBatch length:' + str(batchlength)
    hyperparameters +=  '\nEmbedding dimension:' + str(embeddim) + '\nNumber of embeddings:' + str(numembed)
    hyperparameters += '\nLearning rate:' + str(lr) + '\nLambda:' + str(lam)

    logging.info("Chosen hyperparameters:")
    logging.info(hyperparameters)

    print("Chosen hyperparameters:")
    print(hyperparameters)

    #####################
    #### Train model ####
    #####################
    recon_error, perplex, final_model_name = train_model(datadir, modeldir, fstub, loss, lr=lr, batchsize=batchsize, batchlength=batchlength, quantize=quantize, sparse=sparse, num_embeddings=numembed, embedding_dim=embeddim, lam=lam)
    logging.info("All done training! TOTAL TIME: %s", str(time.time()-prog_start))

    # save losses to file and plot graph!
    results={"reconstruction_error": recon_error, "perplexity": perplex}
    savefile = get_free_filename('recon_error_and_perplexity_' + fstub, modeldir, suffix='.yaml')
    logging.info("SAVING FILE TO: %s", savefile)
    try: 
        with open(savefile, 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)
        save_result_graph(savefile, model_path)
    except Exception as e: 
        print(e)

    ########################
    # Reconstruct tensors! #
    ######################## 
    tensor_dir = Path('..') / 'new_recon_tensors'/ 'train_set_tensors'
    recon_res_dir = Path(modeldir) / 'final_recons'
    if not os.path.isdir(recon_res_dir):
        os.mkdir(recon_res_dir)

    # reconstruct midis
    reconstruct_songs(str(tensor_dir), str(recon_res_dir), str(recon_res_dir), final_model_name, clip_val=0, batchlength=batchlength)
    # Save pianorolls
    save_midi_graphs(str(recon_res_dir),str(recon_res_dir))