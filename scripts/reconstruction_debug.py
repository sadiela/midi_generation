import sys
sys.path.append("..") 
###########
# Imports #
###########
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn.functional as F
from midi_utility import *
#from vq_vae import * 
import matplotlib.pyplot as plt
import pypianoroll
import yaml
from pathlib import Path
import argparse

maxlength = 16*32
from listen_to_model_output import * 

def tensor_to_midi_2(tensor, desired_filepath, bpm=120, subdiv=64, pitchlength_cutoff=0.2):
    # Converts midi tensor back into midi file format
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - tempo = 120bpm
    #   - smallest note subdivision = eighth note (0.250 seconds)
    #   - writes everything as piano 
    # Create new midi object spb = 60/bpm # seconds per beat
    spu = 60/(bpm*subdiv)
    new_mid = pretty_midi.PrettyMIDI() # type=0
    # create a track and add it to the midi
    piano = pretty_midi.Instrument(program=1)
    for pitch in range(tensor.shape[0]):
        max_length=tensor.shape[1]
        note_on=False
        note_start= 0
        current_note_length = 0
        for time in range(tensor.shape[1]):
            if tensor[pitch, time] > 0:
                print(tensor[pitch, time])
            if tensor[pitch,time]>=pitchlength_cutoff and note_on == False:
                note_on=True
                note_start = time
                current_note_length += 1
            elif tensor[pitch,time]>=pitchlength_cutoff and note_on == True:
                current_note_length += 1
            elif tensor[pitch,time]<pitchlength_cutoff and note_on == True:
                # NOTE ENDED
                new_note = pretty_midi.Note(velocity=100, pitch=(pitch), start=(note_start * spu), end=((note_start+current_note_length)*spu))
                piano.notes.append(new_note)                
                note_on=False
                current_note_length=0
            #else: # note on false and tensor[pitch,time]==0
                
    new_mid.instruments.append(piano)

    # save to .mid file 
    new_mid.write(str(desired_filepath))


if __name__ == "__main__":
    tensor_path = '/Users/sadiela/Documents/phd/research/music/midi_generation/recon_debugging/aha__Take_On_Me3_3_conv.npy'
    #model_path = '/Users/sadiela/Documents/phd/research/music/midi_generation/models/model_FINAL-2022-07-01-0.pt'
    cur_tensor = np.load(tensor_path)
    print(cur_tensor)

    '''with open(tensor_path, 'rb') as f:
        pickled_tensor = pickle.load(f)
    cur_tensor = pickled_tensor.toarray()'''

    print(cur_tensor.shape)
    print(np.count_nonzero(cur_tensor))

    print("Values bigger than 10 =", cur_tensor[cur_tensor>0.00456263])
    print("Their indices are ", np.nonzero(cur_tensor > 0.00456263)[0].shape)