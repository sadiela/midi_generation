from music.scripts.midi_utility import PROJECT_DIRECTORY
from vq_vae import *
from gen_utility import * 
import torch
import yaml

#############
# FILEPATHS #
#############
datapath = PROJECT_DIRECTORY + 'midi_data/new_data/midi_tensors/'
modelpath = PROJECT_DIRECTORY + 'models/'
respath = PROJECT_DIRECTORY + 'results/'
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


def main():
    # i think num embeddings was 64 before? 
    model = Model(num_embeddings=1024, embedding_dim=128, commitment_cost=0.5).to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    model_file = get_free_filename('model', modelpath, suffix='.pt')
    recon_error, perplex, nan_recon_files = train_model(datapath, model, model_file)
    # save losses to file
    print("NUM NAN FILES:", len(nan_recon_files))
    results={"reconstruction_error": recon_error, "perplexity": perplex, "nan_reconstruction_files": nan_recon_files}
    savefile = get_free_filename('results', respath, suffix='.yaml')
    with open(savefile, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

if __name__ == "__main__":
    print("START")
    main()