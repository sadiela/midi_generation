      
from pathlib import Path
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('./scripts/')
#import pytest
import numpy as np
import torch
import torch.nn.functional as F
import os

from midi_preprocessing import preprocess
#from vq_vae import DynamicLoss
from dp_loss.sparse_dp_loss import *
import train_vqvae 
import logging

homeDirectory = Path('.')
dataDir = homeDirectory / 'tests' / 'processed_data' / 'sparse_mini'
outDir = homeDirectory / 'tests' / 'results' 
modelDir = homeDirectory / 'tests' / 'models'
testingLog = homeDirectory / 'tests' / 'test_logs' / 'DPTEST1.log'
batchsize = 10
batchlength = 1024
sparse = True
fstub = "DPTEST1"

train_vqvae.train(dataDir, outDir, modelDir, fstub=fstub, loss='dyn', batchsize=batchsize, batchlength=batchlength, normalize=True, quantize=True, sparse=sparse, num_embeddings=1024, embedding_dim=36)
