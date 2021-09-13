import random
import sys
from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL
import os
import pygame
import json
import numpy as np
import pretty_midi

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