import random
import sys
#from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL
import os
import pygame
import json
import numpy as np
import pretty_midi
from tqdm import tqdm 
import re
from glob import glob

from midi_utility import *

#############
# FUNCTIONS #
#############

# WAYS TO GET MORE DATA
'''
- we are trying to create music that sounds cohesive. We can use domain expertise to take some shortcuts and allow our models to train faster/use less data
    - Key: Assume song in the key of Cmaj (or Amin). This limits the # of notes we can use to just 8 PITCH VALUES: 8 POSSIBLE
    - Note length: Assume all notes will be whole notes, half notes, eighth notes, sixteenth notes, or 32nd notes (Only do this if it turns out to be useful)
    - Velocity: assume constant velocity (no dynamic variation)
    - # of tracks: start with single track midis (only one note can play at a time, etc)
'''

############################
# STUFF TO KNOW ABOUT MIDI #
############################
'''
Tempo not given as bpm, given as microseconds per beat
- Default tempo: 500000 microseconds per beat (120bpm)
    - 1 second = 1,000,000 microseconds
    - beats are divided into ticks, ticks are the SMALLEST unit of time in a midi
    - If there are 480tpb, then quarter notes would be 480 ticks long
    - when you give a delta value, you are giving time in ticks
'''

START_DIR = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\lakh\\clean_midi\\The Beatles\\'
#:\Users\sadie\Documents\BU\fall_2021\research\music\midi_data\single_track_midis

########################
# FILE/DIRECTORY PATHS #
########################
#LAKH_DATA_DIR = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\lakh\\clean_midi\\'
DATA_DIR = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\' # '/projectnb/textconv/sadiela/midi_generation/data/' #

SEP_MIDI_DIR = DATA_DIR + 'single_track_midis\\'
SEP_MIDI_DIR_CROPPED = DATA_DIR + 'cropped_midis/'

#CONV_MIDI_DIR = DATA_DIR + 'converted_midis\\'
TENSOR_MIDI_DIR = DATA_DIR + 'midi_tensors/'
#TENSOR_MIDI_DIR_2 = DATA_DIR + 'midi_tensors_2\\'
NORM_TENSOR_MIDI_DIR = DATA_DIR + 'normed_midi_tensors/'

# SEPARATE TRACKS
# Go through all directories in lakh folder
TOP_DIR = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\lakh\\clean_midi\\'
subdirs = [x[0] for x in os.walk(TOP_DIR)]
for subdir in subdirs: 
    print(subdir.split('\\')[-1])
    separate_tracks(subdir, SEP_MIDI_DIR)
#separate_tracks(START_DIR, SEP_MIDI_DIR)

# CROP EMPTY STARTS & CONVERT TO TENSORS
def preprocessing(input_midi_dir, separated=False, cropped=False):
    if not separated: 
        separate_tracks(input_midi_dir, outdir)
    if not cropped: 
        crop_midis(SEP_MIDI_DIR, SEP_MIDI_DIR_CROPPED)
    # convert to tensors
    midis_to_tensors(midi_dirpath, tensor_dirpath, subdiv=32, maxnotelength=16, normalize=False)
    

# Normalize midi tensors
'''maxlength = 16*32
file_list = os.listdir(TENSOR_MIDI_DIR_2)
for file in tqdm(file_list):
    old_tensor = np.load(TENSOR_MIDI_DIR_2 + '\\' + file)
    new_tensor = old_tensor / maxlength 
    if new_tensor.max() > 1.0001:
        print(file, "value larger than 1")
    np.save(NORM_TENSOR_MIDI_DIR_2 + '\\' + file, new_tensor)
bad_chars = [ '(',')','{','}', ' ', "'" ]
for dir, subdir, files in os.walk(NORM_TENSOR_MIDI_DIR_2):
    for file in tqdm(files):
        os.rename(os.path.join(dir,file), os.path.join(dir, "".join(filter(lambda x:x not in bad_chars, file))))
'''


#TENSOR_MIDI_DIR

# LOAD AND PLOT DATA FROM YAML FILE
print("DONE")


#####################
# LOAD IN MIDI FILE #
#####################
# old file
#mid = MidiFile(DATA_DIR + 'classical_piano\\tchaikovsky\\ty_april.mid')
#ableton_mid = MidiFile(simple_scale)
#separate_tracks(LAKH_DATA_DIR + 'ABBA\\', SEP_MIDI_DIR)


#random_midi = "C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\\music\\midi_data\\single_track_midis\\Does Your Mother Know_0.mid"
#generate_random_midi(random_midi, tempo=150)
#play_music(random_midi)
#print("DONE WITH RANDOM")
#print("DONE")
#play_music(dancing_queen)
#input("CONTINUE...")


#numpyfile = DATA_DIR + 'simple_scale_tensor.npy'
#tensor = midi_to_tensor(simple_scale, maxlength=720) # default maxlength is 3 minutes 
#print('SUM OF ALL ELEMENTS:', np.sum(tensor))
#np.save(numpyfile, tensor)

#new_midi_filepath = DATA_DIR + 'speed_up.mid'
#change_tempo(tensor, new_midi_filepath)
# Save tensor to file 
#print(tensor.shape)
#play_music(new_midi_filepath)


'''pretty_scale = pretty_midi.PrettyMIDI(DATA_DIR + 'new_data\\simple_scale.mid')
print('Tempo:', pretty_scale.estimate_tempo())
print('Endtime:', pretty_scale.get_end_time())
print('Instruments:', pretty_scale.instruments)
for instrument in pretty_scale.instruments:
    for note in instrument.notes:
        print(note)
'''

'''input("Continue...")
#play_music(DATA_DIR + 'new_data\\one_note.mid')
#print("NEXT")
#play_music(DATA_DIR + 'new_data\\simple_scale.mid')

#print("DONE PLAYING")


# new midi object 
new_mid = MidiFile() # type=0
# create a track and add it to the midi
new_track = MidiTrack()
new_mid.tracks.append(new_track)

new_track.append(Message('program_change', program=12))

delta = [480]
# 60 = middle C
# KEY OF C MAJOR
notes=[60, 62, 64, 65, 67, 69, 71, 72] #, 73, 74, 75, 76, 77, 78, 79, 80] # 64+7, 64+12]

ticks_per_expr = int(sys.argv[1]) if len(sys.argv) > 1 else 20
for i in range(30):
    note = random.choice(notes)
    delta_val = random.choice(delta)
    new_track.append(Message('note_on', note=note, velocity=127, time=random.choice(delta)))
    #for j in range(delta_val // ticks_per_expr):
    #    pitch = MAX_PITCHWHEEL * j * ticks_per_expr // delta_val
    #    new_track.append(Message('pitchwheel', pitch=pitch, time=ticks_per_expr))
    new_track.append(Message('note_off', note=note, velocity=127, time=0))


# Save new midi file
new_mid.save(DATA_DIR + 'new_data\\one_note.mid')

# Save track as JSON
#with open(DATA_DIR + 'new_data\\ableton.json', 'w') as outfile:
#     json.dump(midifile_to_dict(ableton_mid), outfile, indent=2)

# Save track as JSON
#with open(DATA_DIR + 'new_data\\midi_scale.json', 'w') as outfile:
#     json.dump(midifile_to_dict(ableton_mid), outfile, indent=2)


#print(json.dumps(midifile_to_dict(new_mid), indent=2))


#play_music(DATA_DIR + 'new_data\\ableton_midi.mid')
# Playback the midi
play_music(DATA_DIR + 'new_data\\one_note.mid')
print("NEXT")
#play_music(DATA_DIR + 'classical_piano\\tchaikovsky\\ty_april.mid')


#for track in mid.tracks:
#    print(track)

#print(len(mid.tracks))

#for msg in mid.tracks[1]:
#    print(msg)

##################
# SAVE MIDI FILE #
##################

# Save first two tracks as new file: 
#for track in mid.tracks:
#    print(track)
#mid.save(DATA_DIR + 'new_data\\same_midi.mid')'''


########################
# LISTEN TO MIDI FILES #
########################
