'''
This file contains the VQ-VAE model class
'''
###########
# Imports #
###########
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

# is reconstruction error going down? 
# Run w/ more data
# Increase # of embedding vectors? 

###########################
# MODEL CLASS DEFINITIONS #
###########################
# input: p x t, t variable! p=128
class MIDIVectorQuantizer(nn.Module):
  def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_cost=0.5):
    super().__init__()

    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost

    self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
    self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings) # randomize embeddings to start

  def forward(self, inputs):
    logging.debug("MIDI VECTOR QUANTIZER FORWARD PASS")
    # PASS ONE SONG AT A TIME
    # inputting convolved midi tensors
    # batch dependent on song length, train one song at a time 
    # input = b x p x l

    logging.debug("Original input shape: %s", str(inputs.shape))
    #print("ORIGINAL INPUT SHAPE:",inputs.shape)
    inputs = inputs.squeeze(1) # need to be 2d
    #print("INPUT SHAPE SQUEEZED:",inputs.shape)

    # we will embed along dim p 
    inputs = inputs.permute(0,2,1).contiguous() # now bxlxp
    #print("INPUT SHAPE PERMUTED:",inputs.shape)
    # flatten input
    input_shape = inputs.shape 
    flat_input = inputs.view(-1, self._embedding_dim)
    #(bxl)xp
    logging.debug(flat_input, flat_input.shape)

    distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
    logging.debug("DISTANCES: %s %s", str(distances), str(distances.shape))

    # Encoding
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    #print(encoding_indices)
    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1)
    #print("ENCODINGS:", encodings)
    
    # Quantize and unflatten
    quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
    logging.debug(str(quantized) + str(quantized.shape))

     # Loss
    e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    # e_latent_loss = F.l1_loss(quantized.detach(), inputs)
    q_latent_loss = F.mse_loss(quantized, inputs.detach())
    # q_latent_loss = F.l1_loss(quantized, inputs.detach())
    loss = q_latent_loss + self._commitment_cost * e_latent_loss
    
    quantized = inputs + (quantized - inputs).detach() # backprop through delta
    avg_probs = torch.mean(encodings, dim=0)
    # make sure embeddings are far from each other 
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    
    # convert quantized from 
    return loss, quantized.permute(0, 2, 1).contiguous().unsqueeze(1), perplexity, encodings

class Encoder(nn.Module):
  def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=8,
                                 kernel_size=(1,32))
        self._conv_2 = nn.Conv2d(in_channels=8,
                                 out_channels=4,
                                 kernel_size=(1,64))
        self._conv_3 = nn.Conv2d(in_channels=4,
                                 out_channels=1,
                                 kernel_size=(1,8))
        self.pool = nn.MaxPool2d((1, 2))

  def forward(self, inputs):
          x = self._conv_1(inputs)
          x = F.relu(x)

          x = self._conv_2(x)
          #x = self.pool(x)
          x = F.relu(x)
          x = self._conv_3(x)
          return x

class Decoder(nn.Module):
  def __init__(self, in_channels=1):
        super(Decoder, self).__init__()
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=1,
                                 out_channels=4,
                                 kernel_size=(1,8))
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=4,
                                 out_channels=8,
                                 kernel_size=(1,64))
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=8,
                                 out_channels=1,
                                 kernel_size=(1,32))
        self.pool = nn.MaxPool2d((1, 2))

  def forward(self, inputs):
          print("DECODER INPUT SHAPE:", inputs.shape)
          x = self._conv_trans_1(inputs)

          x = F.relu(x)
          x = self._conv_trans_2(x)
          #x = self.pool(x)
          x = F.relu(x)
          x = self._conv_trans_3(x)
          return x

class VQVAE_Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQVAE_Model, self).__init__()
        
        self._encoder = Encoder(1)
        self._vq_vae = MIDIVectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim)

    def forward(self, x):
      #print("INPUT DIMENSION", x.shape)
      z = self._encoder(x)
      loss, quantized, perplexity, _ = self._vq_vae(z)
      x_recon = self._decoder(quantized)

      return loss, x_recon, perplexity
