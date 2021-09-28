import random
import sys
from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL
import os
import pygame
import json
import numpy as np
import pretty_midi

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

# PLAYBACK CONFIGURATION
# mixer config
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)
pygame.mixer.music.set_volume(0.8) # optional volume 0 to 1.0

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

def play_music(midi_filename):
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

def midifile_to_dict(mid):
    tracks = []
    for track in mid.tracks:
        tracks.append([vars(msg).copy() for msg in track])

    return {
        'ticks_per_beat': mid.ticks_per_beat,
        'tracks': tracks
    }

def generate_random_midi(filepath, num_notes=10, subdivision=-4, tempo=120):
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
    print("POSSIBLE NOTE LENGTHS:", note_lengths)
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

def midi_to_tensor(filepath, maxlength=720): # default maxlength is 3 minutes 
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - tempo = 120bpm
    #   - smallest note subdivision = eighth note (0.250 seconds)
    # returns a 128 x maxlength x 1 tensor representing the midi that was input
    tensor = np.zeros((128,maxlength,1))
    midi = pretty_midi.PrettyMIDI(filepath)
    if len(midi.instruments) > 1:
        print("TOO MANY TRACKS! EMPTY TENSOR RETURNED")
    else:
        for instrument in midi.instruments:
            for note in instrument.notes:
                #print(note.start, note.end, note.pitch, note.velocity)
                #print(note.pitch, int(note.start*4), int(note.start*4 - note.start*4))
                #print(tensor.shape)
                print(int(note.end*4 - note.start*4))
                tensor[note.pitch,int(note.start*4),0] = int(note.end*4 - note.start*4)
                print(tensor[note.pitch,int(note.start*4),0])
    return tensor

def tensor_to_midi(tensor, desired_filepath):
    # Converts midi tensor back into midi file format
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - tempo = 120bpm
    #   - smallest note subdivision = eighth note (0.250 seconds)
    #   - writes everything as piano 
    # Create new midi object
    new_mid = pretty_midi.PrettyMIDI() # type=0
    # create a track and add it to the midi
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    for time in range(tensor.shape[1]):
        for pitch in range(tensor.shape[0]):
            if tensor[pitch,time,0] != 0:
                print("NOTE DETECTED")
                new_note = pretty_midi.Note(velocity=100, pitch=(pitch), start=(time/4), end=((time/4)+(tensor[pitch,time,0]/4)))
                piano.notes.append(new_note)
    new_mid.instruments.append(piano)

    # save to .mid file 
    new_mid.write(desired_filepath)

########################
# FILE/DIRECTORY PATHS #
########################
DATA_DIR = 'C:\\Users\\sadie\\Documents\\fall2021\\research\\music\\midi_generation\\data\\'
simple_scale = DATA_DIR + 'simple_scale.mid'

#####################
# LOAD IN MIDI FILE #
#####################
# old file
#mid = MidiFile(DATA_DIR + 'classical_piano\\tchaikovsky\\ty_april.mid')
ableton_mid = MidiFile(simple_scale)

'''random_midi = DATA_DIR + 'random_midi.mid'
generate_random_midi(random_midi, tempo=150)
play_music(random_midi)
print("DONE WITH RANDOM")'''

play_music(simple_scale)

#numpyfile = DATA_DIR + 'simple_scale_tensor.npy'
#tensor = midi_to_tensor(simple_scale, maxlength=720) # default maxlength is 3 minutes 
#print('SUM OF ALL ELEMENTS:', np.sum(tensor))
#np.save(numpyfile, tensor)

new_midi_filepath = DATA_DIR + 'speed_up.mid'
change_tempo(tensor, new_midi_filepath)
# Save tensor to file 
#print(tensor.shape)
play_music(new_midi_filepath)


'''pretty_scale = pretty_midi.PrettyMIDI(DATA_DIR + 'new_data\\simple_scale.mid')
print('Tempo:', pretty_scale.estimate_tempo())
print('Endtime:', pretty_scale.get_end_time())
print('Instruments:', pretty_scale.instruments)
for instrument in pretty_scale.instruments:
    for note in instrument.notes:
        print(note)
'''
input("Continue...")
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
#mid.save(DATA_DIR + 'new_data\\same_midi.mid')


########################
# LISTEN TO MIDI FILES #
########################

    
#midi_filename = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\music\\midi_data\\classical_piano\\tchaikovsky\\ty_august.mid'

