import random
import sys
from mido import MidiFile, Message, MidiFile, MidiTrack, MAX_PITCHWHEEL
import os
import pygame
import json

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
        'tracks': tracks,
    }

########################
# FILE/DIRECTORY PATHS #
########################
DATA_DIR = 'C:\\Users\\sadie\\Documents\\BU\\fall_2021\\research\music\\midi_data\\'

#####################
# LOAD IN MIDI FILE #
#####################
# old file
mid = MidiFile(DATA_DIR + 'classical_piano\\tchaikovsky\\ty_april.mid')
ableton_mid = MidiFile(DATA_DIR + 'new_data\\ableton_midi.mid')

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
with open(DATA_DIR + 'new_data\\ableton.json', 'w') as outfile:
     json.dump(midifile_to_dict(ableton_mid), outfile, indent=2)


#print(json.dumps(midifile_to_dict(new_mid), indent=2))


play_music(DATA_DIR + 'new_data\\ableton_midi.mid')
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

