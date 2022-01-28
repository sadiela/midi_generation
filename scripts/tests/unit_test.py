'''
Unit tests for VQVAE training workflow
'''
from pathlib import Path
import sys
sys.path.append('..')
#import pytest
import numpy as np
import torch.nn.functional as F

from midi_preprocessing import preprocess
from vq_vae import DynamicLoss
from dp_loss import *

homeDirectory = Path('..')

rawData = homeDirectory / 'tests' / 'raw_data'
procData = homeDirectory / 'tests' / 'test_processed_data'

def testProcessing():
    # clear processed data folder beforehand
    preprocess(rawData, procData)

class DynamicLoss(torch.autograd.Function):
  @staticmethod
  def forward(ctx, recon, data):
    # build theta from original data and reconstruction
    theta = construct_theta(recon, data)
    loss, grad = diffable_recursion(theta)
    ctx.save_for_backward(grad)
    # determine answer
    return loss
  
  @staticmethod
  def backward(ctx):
    grad, = ctx.saved_tensors
    return grad

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


