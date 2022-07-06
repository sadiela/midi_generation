###########
# IMPORTS #
###########
import random
import sys
import os
#import pygame # for playing midi
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pretty_midi # midi manipulation library
from tqdm import tqdm
from gen_utility import * 
from pathlib import Path
from scipy import sparse
import pickle
import json
import re
from glob import glob
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pypianoroll
from midi2audio import FluidSynth
import re
import pygame # for playing midi

from midi_utility import *

##############
# PLAY MUSIC #
##############
# Configure mixer parameters
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples

def play_music(midi_filename):
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(0.8) # optional volume 0 to 1.0
    # plays an entire midi file
    try:
        # use the midi file you just saved
        #Stream music_file in a blocking manner
        clock = pygame.time.Clock()
        pygame.mixer.music.load(midi_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30) # check if playback has finished
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

def play_x_seconds(midi_filename, x=10, savefile=False):
    # play the first x seconds of a midi
    # starts at the time where the first notes starts
    temp_filename = "TEMP_MIDI.MID"
    # create temporary file and "crop"
    crop_midi(midi_filename, temp_filename, cut_beginning=True, maxlength=x)
    play_music(temp_filename)
    #load midi 
    if savefile: 
        print("need to save")
    else: 
        print("delete file when done")
        os.remove(temp_filename)
    # delete temp file


# convert a MIDI file to a wav file
def midi_to_wav(midi_path,wav_path):
        print("CONVERTING")
        # using the default sound font in 44100 Hz sample rate
        fs = FluidSynth()
        fs.midi_to_audio(midi_path, wav_path)