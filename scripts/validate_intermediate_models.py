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
        print(self.paths[index])
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


def validate_model(model, data_path, batchlength= 256, batchsize=5, lossfunc='mse', lam=1, quantize=False):
    midi_tensor_validation = MidiDataset(data_path, l=batchlength) 
    validation_data = DataLoader(midi_tensor_validation, collate_fn=collate_fn, batch_size=batchsize, shuffle=True, num_workers=2)

    model.eval()

    validation_recon_error = []
    validation_total_error = []

    for i, x in enumerate(validation_data):
      x = x.to(DEVICE)

      if quantize: 
            x_hat, vq_loss, perplexity = model(x)
            recon_error = calculate_recon_error(x_hat, x, lossfunc=lossfunc, lam=lam)
            loss = recon_error + vq_loss # will be 0 if no quantization
      else:
            x_hat, mean, log_var = model(x)
            recon_error = calculate_recon_error(x_hat, x, lossfunc=lossfunc, lam=lam) 
            loss = recon_error +  kld(mean, log_var)


      validation_recon_error.append(recon_error.item())
      validation_total_error.append(loss.item())
      if (i+1) % 10 == 0:
        print("Val recon:", recon_error)
        #logging.info('validation recon_error: %.3f' % np.mean(validation_recon_error[-1]))

    model.train()
    print(np.mean(validation_recon_error), np.mean(validation_total_error))

    return validation_recon_error, validation_total_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for running VQ-VAE')
    parser.add_argument('-m', '--modeldir', help='desired model subdirectory name', default='model')
    args = vars(parser.parse_args())

    modelsubdir = args['modeldir']

    modeldir = Path('../models') / modelsubdir

    for file in os.listdir(modeldir):
    # check only text files
        if file.endswith('.pt'):
            print(str(modeldir) + file)
