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
from tqdm import tqdm

#from midi_utility import *

#full_dir ="/Users/sadiela/Documents/phd/research/music/MIDI_data_survey/datasets/lakh_midi/clean_sep_crop_tensors/"
train_dir = "/projectnb/textconv/sadiela/midi_generation/data/ttv/train"
test_dir = "/projectnb/textconv/sadiela/midi_generation/data/ttv/test/"
val_dir = "/projectnb/textconv/sadiela/midi_generation/data//ttv/validate/"

# get list of file names in full_dir
file_list = os.listdir(train_dir)

num_files = len(file_list)
random.shuffle(file_list)

#train_list = file_list[:(num_files//10)*8] # 80 %
val_list =  file_list[(num_files//10)*8: (num_files//10)*9] # 10%
test_list = file_list[(num_files//10)*9:] # 10%

print("UPDATED!")
#print("TRAINING SPLIT:", len(train_list))
#for name in tqdm(train_list): 
 #   shutil.move(full_dir + name , train_dir + name) 

print("VAL SPLIT:", len(val_list), val_dir)
for name in tqdm(val_list): 
    shutil.move(train_dir + name , val_dir + name) 

print("TESTING SPLIT:", len(test_list), test_dir)
for name in tqdm(test_list): 
    shutil.move(train_dir + name , test_dir + name) 

print(len(os.listdir(train_dir)),len(os.listdir(test_dir)),len(os.listdir(val_dir)))