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
import pygame
import json
import numpy as np
import pretty_midi
from tqdm import tqdm 
import re
from glob import glob
import argparse

from midi_utility import *

if __name__ == "__main__":


    print("START")
    midi_directory = PROJECT_DIRECTORY \ 
    separate_tracks(input_midi_dir, out_dir)
    crop_midis(input_midi_dir, out_dir)
    midis_to_tensors(out_dir, midi_dir, subdiv=32, maxnotelength=16, normalize=False)
