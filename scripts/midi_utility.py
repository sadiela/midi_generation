''' 
This file contains general functions that can be used to 
explore and manipulate midi data
'''

###########
# IMPORTS #
###########
import random
import sys
import os
import pygame # for playing midi
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
#import json
import numpy as np
import pretty_midi # midi manipulation library
from tqdm import tqdm
from gen_utility import * 
from pathlib import Path

PROJECT_DIRECTORY = Path('..')
#data_folder = Path("source_data/text_files/")

#####################
# GLOBAL PARAMETERS #
#####################

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

def generate_random_midi(filepath, num_notes=10, subdivision=-4, tempo=120):
    # GENERATE A RANDOM MIDI AND SAVE TO A FILE
    # filepath = location to save new file
    # num_notes = length in notes

    # for now will always write in the key of c, in one octave
    # notes can range from 16th to whole notes
    # notes will not play simultaneously
    # tempo given in bpm
    # assumes 4/4 time 
    # subdivision = =-2 means
    new_mid = pretty_midi.PrettyMIDI() # type=0
    # create a track and add it to the midi
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    quarter_note_duration = tempo/60
    notes=[60, 62, 64, 65, 67, 69, 71, 72] #, 73, 74, 75, 76, 77, 78, 79, 80] # 64+7, 64+12]
    note_lengths = [quarter_note_duration*(2**i) for i in list(range(subdivision, 0)) ]
    #print("POSSIBLE NOTE LENGTHS:", note_lengths)
    last_endtime = 0
    for _ in range(num_notes): 
        cur_pitch = notes[random.randint(0, len(notes)-1)]
        cur_note_length = note_lengths[random.randint(0, len(note_lengths)-1)]
        new_note = pretty_midi.Note(velocity=100, pitch=(cur_pitch), start=last_endtime, end=(last_endtime+cur_note_length))
        piano.notes.append(new_note)
        last_endtime += cur_note_length

    new_mid.instruments.append(piano)

    # save to .mid file 
    new_mid.write(filepath)

def change_tempo(filepath, newfilepath,  maxlength=720, smallest_subdivision=64, target_tempo=120, previous_tempo=None): # default maxlength is 3 minutes 
    # changes tempo of midi file
    
    # get old midi 
    midi = pretty_midi.PrettyMIDI(filepath)
    # old tempo
    if not previous_tempo: 
        previous_tempo = midi.estimate_tempo()
    beat_length = 60/previous_tempo # in seconds secs/beat
    target_beat_length = 60/target_tempo # in seconds secs/beat
    
    # create new midi and instrument 
    new_midi = pretty_midi.PrettyMIDI() # type=0
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # convert notes to proper tempo
    if len(midi.instruments) > 1:
        print("TOO MANY TRACKS! EMPTY TENSOR RETURNED")
    else:
        for instrument in midi.instruments:
            for note in instrument.notes:

                note_length = note.end - note.start 
                number_of_beats = note_length/beat_length

                new_start_time = note.start

                new_note = pretty_midi.Note(velocity=100, pitch=note.pitch, start=note.start*(target_tempo/60)*(target_beat_length), end=(note.start+number_of_beats*target_beat_length))
                piano.notes.append(new_note)
    new_midi.instruments.append(piano)

    # save to .mid file 
    new_midi.write(newfilepath)

####################################
# FUNCTIONS FOR DATA PREPROCESSING #
####################################

def normalize_tensors(orig_tensors, new_dir, subdiv=32, maxnotelength=16):
    maxlength=maxnotelength*subdiv
    file_list = os.listdir(orig_tensors)
    for file in tqdm(file_list):
        cur_tensor = np.load(orig_tensors + file)
        cur_tensor_normed = cur_tensor/maxlength
        if np.count_nonzero == 0: 
            print(cur_tensor.size, np.count_nonzero(cur_tensor)) #, np.count_nonzero(cur_tensor_normed))
        else:
            np.save(new_dir + file.split('.')[0] + '_norm' + '.npy', cur_tensor_normed)


def midis_to_tensors(midi_dirpath, tensor_dirpath, subdiv=32, maxnotelength=16, normalize=False):
    maxlength=maxnotelength*subdiv
    file_list = os.listdir(midi_dirpath)
    for file in tqdm(file_list):
        cur_tensor = midi_to_tensor(midi_dirpath + file, subdiv=subdiv, maxnotelength=maxnotelength)
        if cur_tensor is not None: 
            if not normalize: 
                np.save(tensor_dirpath + file.split('.')[0] + '.npy', cur_tensor)
            else: 
                cur_tensor_normed = cur_tensor/maxlength
                if np.count_nonzero == 0: 
                    print(cur_tensor.size, np.count_nonzero(cur_tensor)) #, np.count_nonzero(cur_tensor_normed))
                else:
                    np.save(tensor_dirpath + file.split('.')[0] + '.npy', cur_tensor_normed)
        else:
            "error in conversion to tensor"

def midi_to_tensor(filepath, subdiv=32, maxnotelength=16): # default maxlength is 3 minutes 
    # maxnotelength given in BEATS
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - smallest note subdivision = 32nd note (0.250 seconds)
    #   - no tempo change
    # returns a 128 x maxlength x 1 tensor representing the midi that was input
    midi_data = pretty_midi.PrettyMIDI(filepath)
    tempo_changes = midi_data.get_tempo_changes() #, midi_data.estimate_tempi())
    if len(tempo_changes[1]) > 1:
        print("TEMPO CHANGES!") # might want to skip file in this situation
    bpm = midi_data.get_tempo_changes()[1][0] #midi_data.estimate_tempo() # in bpm
    bps = bpm/60
    length = midi_data.get_end_time() # in seconds
    tensor_length = (length/60)*bpm*subdiv # # of minutes * beats p minute * beat subdivision
    #print(bpm, bps, length, int(tensor_length))
    tensor = np.zeros((128,int(tensor_length),1))
    if len(midi_data.instruments) > 1:
        print("TOO MANY TRACKS! EMPTY TENSOR RETURNED")
    else:
        try: 
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    note_start = note.start * bps * subdiv
                    # max note length? 
                    note_length = (note.end - note.start) * bps * subdiv
                    #print(note.start, (note.end-note.start), note_start, note_length, round(note_start), round(note_length))
                    tensor[note.pitch,int(note_start),0] = int(min(note_length, subdiv*maxnotelength))
        except Exception as e:
            print("ERROR!", e)
            return None
    return np.squeeze(tensor, axis=2)

def tensors_to_midis(tensor_dir, midi_dir, bpm=120, subdiv=32): 
    # takes a directory of tensors and converts them to midis
    file_list = os.listdir(tensor_dir)
    for file in tqdm(file_list):
        cur_tensor = np.load(tensor_dir + '\\' + file)
        tensor_to_midi(cur_tensor, midi_dir + '\\' + file.split('.')[0] + '.mid')
        

def tensor_to_midi(tensor, desired_filepath, bpm=120, subdiv=32):
    # Converts midi tensor back into midi file format
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - tempo = 120bpm
    #   - smallest note subdivision = eighth note (0.250 seconds)
    #   - writes everything as piano 
    # Create new midi object
    spb = 60/bpm # seconds per beat
    new_mid = pretty_midi.PrettyMIDI() # type=0
    # create a track and add it to the midi
    piano = pretty_midi.Instrument(program=1)
    for time in range(tensor.shape[1]):
        for pitch in range(tensor.shape[0]):
            if tensor[pitch,time] != 0:
                # ADD CUTOFF FOR SHORT NOTES? 
                new_note = pretty_midi.Note(velocity=100, pitch=(pitch), start=(time * (spb/subdiv)), end=((time * (spb/subdiv))+(tensor[pitch,time] * (spb/subdiv))))
                piano.notes.append(new_note)
    new_mid.instruments.append(piano)

    # save to .mid file 
    new_mid.write(desired_filepath)

def separate_tracks(midi_directory, target_directory):
    # takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    for file in file_list: #tqdm(file_list):
        try:
            open_midi = pretty_midi.PrettyMIDI(midi_directory + '\\' + file)
            for i, instrument in enumerate(open_midi.instruments): 
                #print(i, instrument.name, instrument.program, instrument.is_drum)
                cur_midi = pretty_midi.PrettyMIDI() # define new midi object
                cur = pretty_midi.Instrument(program=1) # create new midi instrument
                # copy notes from instrument to cur 
                if not instrument.is_drum:
                    for note in instrument.notes:
                        cur.notes.append(note)
                    # save cur as a new midi file
                    cur_midi.instruments.append(cur)
                    if not os.path.exists(target_directory + file.split('.')[0] + '_'+ str(i) + '.mid'):
                        cur_midi.write(target_directory + file.split('.')[0] + '_'+ str(i) + '.mid')
                    else:
                        print(file.split('.')[0] + '_'+ str(i) + '.mid EXISTS!')
        except Exception as e:
            print("ERROR!", e)
            pass

# Crop all midis in a directory 
def crop_midis(dirname, new_dirname, cut_beginning=True, maxlength=None, remove_special=True): 
    file_list = os.listdir(dirname)
    for file in tqdm(file_list):
        old_name = dirname + file
        if remove_special: 
            new_name = new_dirname + re.sub(r'[^A-Za-z0-9_. ]', r'', file) #remove_special_chars(file) # + file.split('.')[0] + '_cropped.mid'
        else:
            new_name = new_dirname + file
        #if not os.path.exists(new_name):
        crop_midi(old_name, new_name, cut_beginning=cut_beginning, maxlength=maxlength)

def crop_midi(filename, newfilename, cut_beginning=True, maxlength=None):
    # cut out empty space at beginning of midi file
    # maxlength given in seconds
    try:
        open_midi = pretty_midi.PrettyMIDI(filename)
        new_midi = pretty_midi.PrettyMIDI() # define new midi object
        new_instr = pretty_midi.Instrument(program=1) # create new midi instrument
        if cut_beginning:
            start_time = open_midi.instruments[0].notes[0].start
        else: 
            start_time = 0 
        for note in open_midi.instruments[0].notes: 
            if maxlength is not None and note.end-start_time > maxlength:
                break
            else: 
                shifted_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start-start_time, end=note.end-start_time)
                new_instr.notes.append(shifted_note)
        new_midi.instruments.append(new_instr)
        new_midi.write(newfilename)
    except Exception as e: 
        print("Error", e)
        pass
