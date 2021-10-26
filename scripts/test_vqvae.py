from vq_vae import *
import torch

#############
# FILEPATHS #
#############
datapath = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\data\\midi_tensors\\'
modelpath = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\models\\'

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
    model = Model() #num_embeddings, embedding_dim, commitment_cost).to(device)
    train_model(datapath, model, modelpath + 'newmodel.pt')


if __name__ == "__main__":
    print("START")
    main()