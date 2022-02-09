'''
Unit tests for VQVAE training workflow
'''
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

homeDirectory = Path('./scripts/')
print("PATH AND DIRECTORY", sys.path, homeDirectory)

rawData = homeDirectory / 'tests' / 'raw_data'
procData = homeDirectory / 'tests' / 'processed_data'
dataDir = homeDirectory / 'tests' / 'processed_data' / 'sparse_mini'
outDir = homeDirectory / 'tests' / 'results' 
modelDir = homeDirectory / 'tests' / 'models'
testingLog = homeDirectory / 'tests' / 'test_logs' / 'unit_tests.log'
batchsize = 10
sparse = True

#logging.basicConfig(filename=testingLog, level='DEBUG')


trainingParameterList = [ 
    ['l2quant', 'mse', False, True], # filestub, loss type, normalize, quantize
    ['l1quant', 'mae', False, True],
    ['l2', 'mse', False, False],
    ['l1', 'mae', False, False],
    ['l2quantnorm', 'mse', True, True],
    ['l1quantnorm', 'mae', True, True],
    ['l2norm', 'mse', True, False],
    ['l1norm', 'mae', True, False],
    ['dploss', 'dyn', False, False],
    ['dplossquant', 'dyn', False, True]
]

def testProcessing():
    print("PATH AND DIRECTORY", print(os.getcwd()), homeDirectory, sys.path)
    #clear processed data folder beforehand
    preprocess(rawData, procData)

def testTraining():  
  print("Train test")
  for l in trainingParameterList:
      train_vqvae.train(dataDir, outDir, modelDir, fstub=l[0], loss=l[1], batchsize=batchsize, normalize=l[2], quantize=l[3], sparse=sparse)

def testAnalysis():
    print('analysis test')


def testDPLoss():
    mid1 = np.array([
        [1,1,2],
        [0,0,0],
        [0,0,0]
        ])  

    mid2 = np.array([
        [1,1,1],
        [0,1,0],
        [1,0,1]
        ])  

    dyn_loss = dynamic_loss(mid1, mid2)
    print(dyn_loss)


if __name__ == "__main__":
    # try with two example midis:
    mid1 = np.array([
        [1,1,2],
        [0,0,0],
        [0,0,0]
        ])  

    mid2 = np.array([
        [1,1,1],
        [0,1,0],
        [1,0,1]
        ])  

    mid1 = mid1.astype('float64')
    mid2 = mid2.astype('float64')

    mid1 = torch.from_numpy(mid1)
    mid2 = torch.from_numpy(mid2)

    mid2.requires_grad_()

    dynamic_loss = DynamicLoss.apply

    print("L2")
    print(mid1.shape, mid2.shape)
    l2_loss = F.mse_loss(mid1, mid2)
    print(l2_loss)

    print("\nDynamic")
    dyn_loss = dynamic_loss(mid1, mid2)
    print(dyn_loss)
    #dyn_loss.backward()


