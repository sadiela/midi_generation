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

from midi_utility import *

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

    # might not need these two lines
    rawdir = Path(rawdir)
    print(rawdir)
    procdir = Path(procdir) 
    
    # create new directories
    try: 
        os.mkdir(procdir / 'separated') # Do I need another / after to make clear its a directory? probably not 
        os.mkdir(procdir / 'cropped')
        os.mkdir(procdir / 'tensor')
        os.mkdir(procdir / 'sparse')
    except FileExistsError:
        print("Already done!")
        pass

    sep = procdir / 'separated'
    crop = procdir / 'cropped'
    tens = procdir / 'tensor'
    sparse = procdir / 'sparse'
   
    print("START2")
    separate_tracks(rawdir, sep)
    crop_midis(sep, crop)
    midis_to_tensors(crop, tens, subdiv=32, maxnotelength=16, normalize=False)
    convert_to_sparse(tens, sparse, del_tensor_dir=True)
    # empty other dirs

    for f in os.listdir(sep):
            os.remove(os.path.join(sep, f))

    for f in os.listdir(crop):
            os.remove(os.path.join(crop, f))

