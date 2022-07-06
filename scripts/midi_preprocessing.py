'''
Preprocessing for new midi files:
- separating tracks
- cropping empty starts
- convert to tensors
- save tensors sparsely
'''
import random
import sys
import os
import json
import numpy as np
import pretty_midi
from tqdm import tqdm 
import re
from glob import glob
import argparse
from pathlib import Path
import shutil

from midi_utility import *


def train_test_split(full_dir):
    # create new directories
    try: 
        os.mkdir(full_dir / 'test')
        os.mkdir(full_dir / 'train')
        os.mkdir(full_dir / 'validate')
    except FileExistsError:
        print("Already done!")
        pass

    test_dir = full_dir / 'test'
    train_dir = full_dir / 'train'
    val_dir = full_dir / 'validate'

    # get list of file names in full_dir
    file_list = os.listdir(full_dir)

    num_files = len(file_list)
    random.shuffle(file_list)

    train_list = file_list[:(num_files//10)*8] # 80 %
    val_list =  file_list[(num_files//10)*8: (num_files//10)*9] # 10%
    test_list = file_list[(num_files//10)*9:] # 10%

    print("TRAINING SPLIT:", len(train_list))
    for name in tqdm(train_list): 
       shutil.move(full_dir + name , train_dir + name) 

    print("VAL SPLIT:", len(val_list), val_dir)
    for name in tqdm(val_list): 
        shutil.move(train_dir + name , val_dir + name) 

    print("TESTING SPLIT:", len(test_list), test_dir)
    for name in tqdm(test_list): 
        shutil.move(train_dir + name , test_dir + name) 

    print(len(os.listdir(train_dir)),len(os.listdir(test_dir)),len(os.listdir(val_dir)))

def preprocess(rawdir, procdir):
    # might not need these two lines
    rawdir = Path(rawdir)
    print(rawdir)
    procdir = Path(procdir) 
    
    # create new directories
    try: 
        os.mkdir(procdir / 'cropped')
        os.mkdir(procdir / 'sparse')
    except FileExistsError:
        print("Already done!")
        pass

    crop = procdir / 'cropped'
    sparse = procdir / 'sparse'
   
    print("START2")
    #separate_tracks(rawdir, sep)
    #crop_midis(sep, crop)
    sep_and_crop(rawdir, crop)
    midis_to_tensors_2(crop, sparse, subdiv=32, maxnotelength=256, normalize=False)
    # empty other dirs

    for f in os.listdir(crop):
            os.remove(os.path.join(crop, f))

    train_test_split(sparse)

if __name__ == "__main__":
    print("START!")
    datapath = PROJECT_DIRECTORY  / 'data' / 'raw_data'  # '..\\midi_data\\full_dataset_midis_normalized\\'
    processed_datapath = PROJECT_DIRECTORY / 'data' / 'preprocessed_data'

    parser = argparse.ArgumentParser(description='Arguments for preprocessing midis')
    #parser.add_argument('-l','--lossfunc', help='Choose loss function.', default=True)
    parser.add_argument('-r', '--rawdata', help='Path to raw MIDI data.', default=datapath)
    parser.add_argument('-p', '--pdata', help='Path where processed data will be saved', default=processed_datapath)


    args = vars(parser.parse_args())

    rawdir = args['rawdata']
    procdir = args['pdata']

    preprocess(rawdir, procdir)