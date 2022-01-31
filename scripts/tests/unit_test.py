'''
Unit tests for VQVAE training workflow
'''
from pathlib import Path
import sys
sys.path.append('..')
sys.path.append('.')
#import pytest
import numpy as np
import torch.nn.functional as F

from midi_preprocessing import preprocess
from vq_vae import DynamicLoss
from dp_loss import *
import train_vqvae 
import logging

homeDirectory = Path('..')

rawData = homeDirectory / 'tests' / 'raw_data'
procData = homeDirectory / 'tests' / 'test_processed_data'
dataDir = homeDirectory / 'tests' / 'processed_data' / 'sparse'
outDir = homeDirectory / 'tests' / 'results' 
modelDir = homeDirectory / 'tests' / 'models'
testingLog = homeDirectory / 'tests' / 'test_logs' / 'unit_tests.log'
batchsize = 10
sparse = True

logging.basicConfig(filename=testingLog, encoding='utf-8', level='DEBUG')


trainingParameterList = [ 
    ['l2quant', 'mse', False, True],
    ['l1quant', 'mae', False, True],
    ['l2', 'mse', False, False],
    ['l1', 'mae', False, False],
    ['l2quantnorm', 'mse', True, True],
    ['l1quantnorm', 'mae', True, True],
    ['l2norm', 'mse', True, False],
    ['l1norm', 'mae', True, False],
]

def testProcessing():
    # clear processed data folder beforehand
    preprocess(rawData, procData)

def testTraining():  
  print("Train test")
  for l in trainingParameterList:
      train_vqvae.train(dataDir, outDir, modelDir, fstub=l[0], loss=l[1], batchsize=batchsize, normalize=l[2], quantize=l[3], sparse=sparse)

def testAnalysis():
    print('analysis test')

if __name__ == "__main__":
    # try with two example midis:
    mid1 = np.array([
        [0,0,0,0,0,2],
        [2,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,5,0,1,0,2],
        [0,0,0,0,0,0]
        ])  

    mid2 = np.array([
        [1,0,0,0,0,0],
        [0,0,2,0,0,0],
        [0,2,0,0,0,0],
        [0,0,0,1,0,2],
        [0,0,0,0,0,0]
        ])  

    mid1 = mid1.astype('float64')
    mid2 = mid2.astype('float64')


    dynamic_loss = DynamicLoss.apply

    print("L2")
    print(mid1.shape, mid2.shape)
    l2_loss = F.mse_loss(torch.Tensor(mid1), torch.Tensor(mid2))
    print(l2_loss.grad)
    l2_loss.backward()
    print(l2_loss.grad)
    print(l2_loss)

    print("\nDynamic")
    dyn_loss = dynamic_loss(mid1, mid2)
    print(dyn_loss.grad)
    dyn_loss.backward()
    print(dyn_loss.grad)
    print(dyn_loss)


