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

dynamic_loss = DynamicLoss.apply

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

def compare_losses(mid1, mid2): 
    l2_loss = F.mse_loss(mid1, mid2)
    dyn_loss = dynamic_loss(mid1, mid2)
    print("L2:", l2_loss.data)
    print("Dynamic:", dyn_loss.data)


if __name__ == "__main__":
    # try with two example midis:
    mid1 = torch.tensor([
        [1,1],
        [0,0],
        ], dtype=torch.float32)  

    mid2 = torch.tensor([
        [1,1],
        [0,1]
        ], dtype=torch.float32)  

    mid3 = torch.tensor([
        [1,1],
        [1,1],
        ], dtype=torch.float32)  

    mid4 = torch.tensor([
        [1,0],
        [0,0]
        ], dtype=torch.float32)  

    mid5 = torch.tensor([
        [1,2],
        [0,0],
        ], dtype=torch.float32)  

    mid6 = torch.tensor([
        [1,4],
        [0,1]
        ], dtype=torch.float32)  

    mid6 = torch.tensor([
        [0,0],
        [0,0]
        ], dtype=torch.float32)  

    mid2.requires_grad_()
    mid3.requires_grad_()
    mid4.requires_grad_()
    mid5.requires_grad_()
    mid6.requires_grad_()


    compare_losses(mid1, mid2)
    compare_losses(mid1, mid3)
    compare_losses(mid1, mid4)  
    compare_losses(mid1, mid5)
    compare_losses(mid1, mid6)  
    #compare_losses(mid1, mid7)
    #compare_losses(mid1, mid8)  


