      
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
from vq_vae import DynamicLoss
from dp_loss import *
import train_vqvae 
import logging

homeDirectory = Path('.')
dataDir = homeDirectory / 'tests' / 'processed_data' / 'sparse_mini'
outDir = homeDirectory / 'tests' / 'results' 
modelDir = homeDirectory / 'tests' / 'models'
testingLog = homeDirectory / 'tests' / 'test_logs' / 'unit_tests.log'
batchsize = 10
sparse = True
fstub = "dpTEST"

train_vqvae.train(dataDir, outDir, modelDir, fstub=fstub, loss='dyn', batchsize=batchsize, normalize=True, quantize=True, sparse=sparse)
