# split all data into train/test/val

import random
import sys
import os
import json
from tqdm import tqdm 
import re
from glob import glob
import argparse
from pathlib import Path
import shutil

#from midi_utility import *

full_dir = '/projectnb/textconv/sadiela/midi_generation/data/all_midi_tensors_sparse/'
train_dir = '/projectnb/textconv/sadiela/midi_generation/data/all_midi_tensors_ttv/'
test_dir = '/projectnb/textconv/sadiela/midi_generation/data/all_midi_tensors_ttv/'
val_dir = '/projectnb/textconv/sadiela/midi_generation/data/all_midi_tensors_ttv/'

# get list of file names in full_dir
file_list = os.listdir(full_dir)

num_files = len(file_list)
random.shuffle(file_list)

train_list = file_list[:(num_files//10)*8] # 80 %
val_list =  file_list[(num_files//10)*8: (num_files//10)*9] # 10%
test_list = file_list[(num_files//10)*9:] # 10%

print("TRAINING SPLIT:", len(train_list))
for name in tqdm(train_list): 
    shutil.copy2(full_dir + name , train_dir + name) 

print("VAL SPLIT:", len(val_list))
for name in tqdm(val_list): 
    shutil.copy2(full_dir + name , val_dir + name) 

print("TESTING SPLIT:", len(test_list))
for name in tqdm(test_list): 
    shutil.copy2(full_dir + name , test_dir + name) 