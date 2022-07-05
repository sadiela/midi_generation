''' 
This file contains general functions that can be used to 
explore and manipulate midi data as well as preprocessing functions
'''

###########
# IMPORTS #
###########
import random
import os
#import pygame # for playing midi
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import pretty_midi # midi manipulation library
from tqdm import tqdm
from gen_utility import * 
from scipy import sparse
import re
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pypianoroll
import re

#####################
# GLOBAL PARAMETERS #
#####################
PROJECT_DIRECTORY = Path('..')

##############
# PLAY MUSIC #
##############
# Configure mixer parameters
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples

keys_to_notes = {
    0: [0,2,4,5,7,9,11],
    1: [0,1,2,5,6,8,10],
    2: [1,2,4,6,7,9,11],
    3: [0,2,3,5,7,8,10],
    4: [1,3,4,6,8,9,11], 
    5: [1,3,5,6,8,10,11], 
    6: [0,2,4,5,7,9,10], 
    7: [0,1,3,5,7,8,10],
    8: [0,2,4,6,7,9,11],
    9: [0,2,3,5,7,9,10],
    10: [1,2,4,6,8,9,11],
    11: [1,3,4,6,8,10,11]
}

key_number_to_name= {
    0:"C",
    1:"C#/Db",
    2:"D",
    3:"D#/Eb",
    4:"E",
    5:"F",
    6:"F#/Gb",
    7:"G",
    8:"G#/Ab",
    9:"A",
    10:"A#/Bb",
    11:"B",
}

'''def play_music(midi_filename):
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




'''

# convert a MIDI file to a wav file
from midi2audio import FluidSynth
def midi_to_wav(midi_path,wav_path):
        print("CONVERTING")
        # using the default sound font in 44100 Hz sample rate
        fs = FluidSynth()
        fs.midi_to_audio(midi_path, wav_path)

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


#### DON'T THINK THIS IS BEING USED ANYMORE...###
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
        cur_tensor = midi_to_tensor(midi_dirpath / file, subdiv=subdiv, maxnotelength=maxnotelength)
        if cur_tensor is not None: 
            if not normalize: 
                np.save(tensor_dirpath / str(file.split('.')[0] + '.npy'), cur_tensor)
            else: 
                cur_tensor_normed = cur_tensor/maxlength
                if np.count_nonzero == 0: 
                    print(cur_tensor.size, np.count_nonzero(cur_tensor)) #, np.count_nonzero(cur_tensor_normed))
                else:
                    np.save(tensor_dirpath / str(file.split('.')[0] + '.npy'), cur_tensor_normed)
        else:
            "error in conversion to tensor"

def midis_to_tensors_2(midi_dirpath, tensor_dirpath, subdiv=64, maxnotelength=256, normalize=False):
    maxlength=maxnotelength*subdiv
    file_list = os.listdir(midi_dirpath)
    for file in tqdm(file_list):
        if not os.path.exists(tensor_dirpath / str(file.split('.')[0] + '.p')):
            cur_tensor = midi_to_tensor_2(midi_dirpath / file, subdiv=subdiv, maxnotelength=maxnotelength)
            if cur_tensor is not None: 
                if not normalize: 
                    sparse_arr = sparse.csr_matrix(cur_tensor) # save sparse!!!
                    with open(tensor_dirpath / str(file.split('.')[0] + '.p'), 'wb') as outfile:
                        pickle.dump(sparse_arr, outfile)
                else: 
                    cur_tensor_normed = cur_tensor/maxlength
                    if np.count_nonzero == 0: 
                        print(cur_tensor.size, np.count_nonzero(cur_tensor)) #, np.count_nonzero(cur_tensor_normed))
                    else:
                        np.save(tensor_dirpath / str(file.split('.')[0] + '.npy'), cur_tensor_normed)
            else:
                "error in conversion to tensor"
        #else:
        #    print("Already converted", file)

def midi_to_tensor(filepath, subdiv=32, maxnotelength=16): # default maxlength is 3 minutes 
    # maxnotelength given in BEATS
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - smallest note subdivision = 32nd note (0.250 seconds)
    #   - no tempo change
    # returns a 128 x maxlength x 1 tensor representing the midi that was input
    midi_data = pretty_midi.PrettyMIDI(str(filepath))
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
    new_mid.write(str(desired_filepath))

def separate_tracks(midi_directory, target_directory):
    # takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    for file in file_list: #tqdm(file_list):
        try:
            open_midi = pretty_midi.PrettyMIDI(str(midi_directory / file))
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
                    if not os.path.exists(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')):
                        cur_midi.write(str(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')))
                    else:
                        print(file.split('.')[0] + '_'+ str(i) + '.mid EXISTS!')
        except Exception as e:
            print("ERROR!", e)
            pass

# Crop all midis in a directory 
def crop_midis(dirname, new_dirname, cut_beginning=True, maxlength=None, remove_special=True): 
    file_list = os.listdir(dirname)
    for file in tqdm(file_list):
        old_name = dirname / file
        if remove_special: 
            new_name = new_dirname / re.sub(r'[^A-Za-z0-9_. ]', r'', file) #remove_special_chars(file) # + file.split('.')[0] + '_cropped.mid'
        else:
            new_name = new_dirname / file
        #if not os.path.exists(new_name):
        crop_midi(old_name, new_name, cut_beginning=cut_beginning, maxlength=maxlength)

def crop_midi(filename, newfilename, cut_beginning=True, maxlength=None):
    # cut out empty space at beginning of midi file
    # maxlength given in seconds
    try:
        open_midi = pretty_midi.PrettyMIDI(str(filename))
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
        new_midi.write(str(newfilename))
    except Exception as e: 
        print("Error", e)
        pass

def sep_and_crop(midi_directory, target_directory):
# takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    for file in tqdm(file_list): #tqdm(file_list):
        try:
            open_midi = pretty_midi.PrettyMIDI(str(midi_directory / file))
            for i, instrument in enumerate(open_midi.instruments): 
                #print(i, instrument.name, instrument.program, instrument.is_drum)
                cur_midi = pretty_midi.PrettyMIDI() # define new midi object
                cur_inst = pretty_midi.Instrument(program=1) # create new midi instrument
                # copy notes from instrument to cur 
                if not instrument.is_drum:
                    start_time = instrument.notes[0].start
                    for note in instrument.notes:
                        shifted_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start-start_time, end=note.end-start_time)
                        cur_inst.notes.append(shifted_note)
                    # save cur as a new midi file
                    cur_midi.instruments.append(cur_inst)
                    if not os.path.exists(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')):
                        cur_midi.write(str(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')))
                    else:
                        print(file.split('.')[0] + '_'+ str(i) + '.mid EXISTS!')
        except Exception as e:
            print("ERROR!", e)
            pass

def convert_to_sparse(tensor_dir, sparse_dir, del_tensor_dir=False):
    file_list = os.listdir(tensor_dir)
    for f in tqdm(file_list): 
        cur_arr = np.load(tensor_dir / f)
        new_f = f.split('.')[0] + '.p'
        sparse_arr = sparse.csr_matrix(cur_arr)
        with open(sparse_dir / new_f, 'wb') as outfile:
                pickle.dump(sparse_arr, outfile)
    # Delete tensor_dir to save space
    if del_tensor_dir: 
        for f in os.listdir(tensor_dir):
            os.remove(os.path.join(tensor_dir, f))
    print("DONE!")


def change_midi_key(old_midi_path, new_midi_path):
    if not Path(new_midi_path).is_file():    
        try: 
            open_midi = pretty_midi.PrettyMIDI(old_midi_path)
            total_velocity = sum(sum(open_midi.get_chroma())) # collapses pianoroll accross octaves

            semitone_velocities = np.array([sum(semitone)/total_velocity for semitone in open_midi.get_chroma()]) # assuming C is 0?

            ind = np.argpartition(semitone_velocities, -7)[-7:]

            key=0
            for i in range(12):
                if set(keys_to_notes[i]) == set(ind):
                    #print("KEY:", key_number_to_name[i], i)
                    key = i
                    break

            # shift piece into the key of c
            if key != 0:
                for i, instrument in enumerate(open_midi.instruments): 
                        if not instrument.is_drum:
                            for note in instrument.notes:
                                note.pitch -= key

            # check if file exists
            open_midi.write(new_midi_path)
        except Exception as e:
            print("ERROR!", e, str(old_midi_path))
    else:
        print("FILE EXISTS:", new_midi_path)

### CODE FOR NEW MIDI REPRESENTATION ###
def midi_to_tensor_2(filepath, subdiv=64, maxnotelength=256): # default maxlength is 3 minutes 
    # maxnotelength given in BEATS
    # ASSUMES:
    #   - 1 track
    #   - constant note velocity (100)
    #   - smallest note subdivision = 32nd note (0.250 seconds)
    #   - no tempo change
    # returns a 128 x maxlength x 1 tensor representing the midi that was input
    try: 
        midi_data = pretty_midi.PrettyMIDI(str(filepath))
        tempo_changes = midi_data.get_tempo_changes() #, midi_data.estimate_tempi())
        if len(tempo_changes[1]) > 1:
            print("TEMPO CHANGES!") # might want to skip file in this situation
        bpm = midi_data.get_tempo_changes()[1][0] #midi_data.estimate_tempo() # in bpm
        bps = bpm/60
        midi_length = midi_data.get_end_time() # in seconds
        tensor_length = midi_length*bps*subdiv #(length/60)*bpm*subdiv # # of minutes * beats p minute * beat subdivision
        # declare tensor
        tensor = np.zeros((128,int(tensor_length),1))
        if len(midi_data.instruments) > 1:
            print("TOO MANY TRACKS! EMPTY TENSOR RETURNED")
        else:

                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        note_start = note.start * bps * subdiv
                        note_end = note.end * bps * subdiv
                        note_length = note_end - note_start
                        #print(note_start, note_end, note_length)
                        tensor[note.pitch, int(note_start):int(note_start+note_length-1),0]=1
    except Exception as e:
        print("ERROR!", e)
        return None
    return np.squeeze(tensor, axis=2)

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

def change_keys_and_names(orig_midi_dir, new_midi_dir):
    all_dirs = os.listdir(orig_midi_dir)
    for d in tqdm(all_dirs): 
        files = os.listdir(orig_midi_dir / d)
        for f in files: 
            new_midi_name = re.sub(r'[^\w\s]', '', d + '__' + f[:-4]).replace(" ", "_") + ".mid"
            change_midi_key(str(orig_midi_dir / d / f), str(new_midi_dir / new_midi_name))

def show_graph(midi_path):
    recon = pypianoroll.read(midi_path)
    recon.trim(0, 34*recon.resolution)
    recon.plot()
    plt.title("OLD MIDI")
    plt.show()


if __name__ == "__main__":
    #orig_midi_dir = Path("/Users/sadiela/Documents/phd/research/music/MIDI_data_survey/datasets/lakh_midi/clean_midi_c_key")
    #sep_crop_dir = Path("/Users/sadiela/Documents/phd/research/music/MIDI_data_survey/datasets/lakh_midi/clean_midi_c_sep_crop")
    #tensor_dir = Path("/Users/sadiela/Documents/phd/research/music/MIDI_data_survey/datasets/lakh_midi/clean_sep_crop_tensors")
    mid_path = "/Users/sadiela/Documents/phd/research/music/midi_generation/scripts/AllYouNeedIsLove_11_cropped.mid"
    wav_path = "/Users/sadiela/Documents/phd/research/music/midi_generation/scripts/AllYouNeedIsLove_11_cropped.wav"
    
    new_mid = pretty_midi.PrettyMIDI() # type=0
    print(new_mid.get_end_time())
    #print("Separating tracks and cropping empty starts")
    #sep_and_crop(orig_midi_dir, sep_crop_dir)

    #old_midi_path = "/Users/sadiela/Documents/phd/research/music/MIDI_data_survey/datasets/lakh_midi/clean_midi_c_sep_crop/Londonbeat__Ive_Been_Thinking_About_You2_2.mid"
    #new_midi_path = "/Users/sadiela/Documents/phd/research/music/MIDI_data_survey/datasets/lakh_midi/midi_to_tensor_test/new_midi.mid"

    #old_midi = pretty_midi.PrettyMIDI(str(orig_midi_dir/midi_name))
    #new_midi = pretty_midi.PrettyMIDI(new_midi_path)
    #wav_form = open_midi.synthesize()
    #old_piano_roll = old_midi.get_piano_roll()
    #new_piano_roll = new_midi.get_piano_roll()
    #print(old_piano_roll.shape, new_piano_roll.shape)


    #f, axarr = plt.subplots(2,1)
    #plt.imshow(old_piano_roll[40:-40,:1000], interpolation='none')
    '''plt.figure(figsize=(10,10))
    plt.imshow(new_piano_roll[40:-40,:1000], interpolation='none')
    plt.show()'''

    #midis_to_tensors_2(sep_crop_dir, tensor_dir, subdiv=64, maxnotelength=256, normalize=False)

    '''
    all_files = os.listdir(sep_crop_dir)
    for file in all_files:
        midi_path = sep_crop_dir / file 
        #orig_wav = out_dir / "orig_wav.wav"
        #midi_to_wav(midi_path,orig_wav)
        print(midi_path)
        input("Continue...")
        tensor_version = midi_to_tensor_2(midi_path, subdiv=64, maxnotelength=256)
        desired_filepath = out_dir / "new_midi.mid"
        desired_wav_filepath = out_dir / "new_wav.wav"
        back_to_midi = tensor_to_midi_2(tensor_version, desired_filepath, bpm=120, subdiv=64)
        midi_to_wav(desired_filepath,desired_wav_filepath)
        input("Continue...")'''
