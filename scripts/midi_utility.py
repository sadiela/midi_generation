''' 
This file contains general functions that can be used to 
explore and manipulate midi data
'''

###########
# IMPORTS #
###########
import random
import sys
from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL
import os
import pygame # for playing midi
import json
import numpy as np
import pretty_midi # midi manipulation library

#####################
# GLOBAL PARAMETERS #
#####################
temp_file_dir = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\data\\temp_files\\'
crop_file_dir = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\data\\cropped_midis\\'
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

# COMPLETE IMPLEMENTATION!!!! #
def play_x_seconds(midi_filename, x, savefile=False):
    # play the first x seconds of a midi
    # starts at the time where the first notes starts

    # create temporary file and "crop"
    if savefile: 
        print("need to save")
    else: 
        print("delete file when done")

    # play temp file
    play_music(temp_file)

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

####################################
# FUNCTIONS FOR DATA PREPROCESSING #
####################################

def midi_to_tensor(filepath, subdiv=32): # default maxlength is 3 minutes 
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
    print(bpm, bps, length, int(tensor_length))
    tensor = np.zeros((128,int(tensor_length),1))
    if len(midi_data.instruments) > 1:
        print("TOO MANY TRACKS! EMPTY TENSOR RETURNED")
    else:
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note_start = note.start * bps * subdiv
                note_length = (note.end - note.start) * bps * subdiv
                #print(note.start, (note.end-note.start), note_start, note_length, round(note_start), round(note_length))
                tensor[note.pitch,int(note_start),0] = int(note_length)
    return np.squeeze(tensor, axis=2)

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
                new_note = pretty_midi.Note(velocity=100, pitch=(pitch), start=(time * (spb/subdiv)), end=((time * (spb/subdiv))+(tensor[pitch,time] * (spb/subdiv))))
                piano.notes.append(new_note)
    new_mid.instruments.append(piano)

    # save to .mid file 
    new_mid.write(desired_filepath)

def separate_tracks(midi_directory, target_directory):
    # takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    for file in file_list:
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
                    cur_midi.write(target_directory + file.split('.')[0] + '_'+ str(i) + '.mid')
        except Exception as e:
            print("ERROR!", e)
            pass

def crop_midi(filename, newfilename):
    # cut out empty space at beginning of midi file
    try:
        open_midi = pretty_midi.PrettyMIDI(filename)
        new_midi = pretty_midi.PrettyMIDI() # define new midi object
        new_instr = pretty_midi.Instrument(program=1) # create new midi instrument
        start_time = open_midi.instruments[0].notes[0].start  
        for note in open_midi.instruments[0].notes: 
            shifted_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start-start_time, end=note.end-start_time)
            new_instr.notes.append(shifted_note)
        new_midi.instruments.append(new_instr)
        new_midi.write(newfilename)
    except Exception as e: 
        print("Error", e)
        pass
