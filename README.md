# Midi Generation
Research project started summer 2021.

## Goal
Generate cohesive midis possessing the structure and repetition found in most modern music.

## High Level Approach
We will use an approach inspired by the Jukebox model.

## Set-up/Dependencies
This package has the followng dependencies:
* python
* pytorch
* numpy
* pretty_midi
* pygame
* tqdm
* yaml

You can set up your virtual environment and download all of these dependencies with the following commands:

    # create virtual environment (replace myenv with a name of your choosing)
    conda create -n myenv python=3.8.12 pygame tqdm yaml pretty_midi numpy pytorch scipy

    # activate your new virtual environment
    conda activate myenv

This model is fairly large. It is desirable to train on a machine with GPUs for reasonable running time.

## Usage:
### `midi_utility.py`

This file contains most of the core functions for manipulating MIDI data.

#### Functons that work with MIDI input
* `sep_and_crop(midi_directory, target_directory)`: Takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track in the original files, so each output midi has a single track; crops the beginning of a track up until the first note. 
* `midis_to_tensors_2(midi_dirpath, tensor_dirpath, subdiv)`: Provide the path to a directory full of MIDIs; converts all the MIDIs to tensors and saves them in the provided tensor directory. 
* `midi_to_tensor_2(filepath, subdiv)`: Subroutine of `midis_to_tensors` that performs conversion of MIDI to tensor for one MIDI file; returns the tensor.
* `change_keys_and_names(orig_midi_dir, new_midi_dir)`: Specialty function I used on the `clean_lakh` data to remove special characters from song names and prepend the artist name to each song name. This function also changes the key of all the MIDIs provided and saves them to the new directory. 
* `change_midi_key(old_midi_path, new_midi_path, desired_key=0)`: Input the path to a MIDI, the path where you want your new MIDI to be saved, and the desired key (key of C=0, etc.). Determines the key of the MIDI and shifts all the notes to the desired key. Works best on multi-track MIDIs (pre-separation) because you have more data with which to determine the key. 
* `show_graph(midi_path)`: Display a plot of the pianoroll for the provided MIDI

##### Functions that work with tensor input
* `tensors_to_midis_2(tensor_dir, midi_dir, subdiv)`: Provide the path to a directory full of tensors (stored as pickle files); converts all the tensors to MIDIs and saves them in the provided MIDI directory. 
* `tensor_to_midi_2(filepath, subdiv)`: Subroutine of `tensors_to_midis_2` that performs conversion of tensor to MIDI for one tensor file; saves the MIDI to the desired filepath.


### `midi_preprocessing.py`

Preprocessing steps. Starting with a directory of midi files, executes the following:

Separates midis into individual tracks (for now)
* Crops empty beginnings from separated midis
* Converts midis to tensor representations
* Stores tensors in sparse format to save memory
* Splits into 80/10/10 train/test/validate directories

To run: provide the following command line arguments:
* -r: raw data directory
* -p: directory where you want the processed data to be stored

**(would be nice to add key change code to this pipeline!)**

`python midi_preprocessing -r RAW_DATA_DIR -p PROCESSED_DATA_DIR`

### `train_vqvae.py`
To train the VAE/VQ-VAE model:

* Clone repository to your local machine
* Navigate to /scripts directory
* Run python test_vqvae.py with the following arguments:
    * -d: path to data directory (midi tensors)
    * -m: path where you want model/results saved
    * -r: result and model filestub
    * -q: VAE or VQ-VAE (defaults to VQ-VAE)
    * -b: number of songs in a batch
    * -a: Length of a batch (want to change to 384 once subdiv is 96)
    * -u: Number of embeddings (for VQ-VAE)
    * -e: Embedding dimension
    * -l: specify loss function (default is MSE, VAE always uses BCEloss)
    * -v: increase logging verbosity

### `reconstruct_from_model.py`

This file contains various functions for analyzing trained model results. You can do the following:

* `reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path)`: reconstruct a directory of midi tensors using the VQ-VAE model; save as tensors and midis
* `reconstruct_song(orig_tensor_path, model)`: subroutine of `reconstruct_songs`; returns a reconstruction and loss for a provided tensor.
* `save_midi_graphs(midi_path, save_path)`: save pianoroll images of midis
* `save_result_graph(yaml_file, plot_dir)`: save reconstruction error and perplexity graphs over model training
