from vq_vae import *
import torch
import yaml

#############
# FILEPATHS #
#############
datapath = '/usr3/graduate/sadiela/midi_generation/data/normed_midi_tensors_2/'
modelpath = '/usr3/graduate/sadiela/midi_generation/models/' #model_10_25_2.pt'
respath = '/results/'

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
    model = Model().to(device) #num_embeddings, embedding_dim, commitment_cost).to(device)
    recon_error, perplex = train_model(datapath, model, modelpath + 'newmodel.pt')
    # save losses to files
    results={"reconstruction_error": recon_error, "perplexity": perplex}
    with open(respath + 'res.yml', 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

if __name__ == "__main__":
    print("START")
    main()