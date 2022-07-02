## Midi Generation
Research project started summer 2021.

### Goal
Generate cohesive midis possessing the structure and repetition found in most modern music.

### High Level Approach
We will use an approach inspired by the Jukebox model.

### Set-up/Dependencies
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

### Usage:
`midi_preprocessing.py`
Preprocessing steps. Starting with a directory of midi files, executes the following:

Separates midis into individual tracks (for now)
* Crops empty beginnings from separated midis
* Converts midis to tensor representations
* Stores tensors in sparse format to save memory
* To run: provide the following command line arguments:
    * -r: raw data directory
    * -p: directory where you want the processed data

`python midi_preprocessing -r RAW_DATA_DIR -p PROCESSED_DATA_DIR`

`train_vqvae.py`
To train the VQ-VAE model:

* Clone repository to your local machine
* Navigate to /scripts directory
* Run python test_vqvae.py with the following (optional) arguments:(
    * -d: path to data directory (midi tensors)
    * -m: path where you want model saved
    * -o: path to desired results directory
    * -r: result and model filestub
    * -n: whether or not to normalize the tensors (default is not to normalize)
    * -b: number of songs in a batch
    * -l: specify loss function (default is MSE, using flag switches to MAE)
    * -v: increase logging verbosity
    * -q: whether or not to quantize representations

`listen_to_model_output.py`
This file contains various functions for analyzing trained model results. You can do the following:

* `reconstruct_songs(orig_tensor_dir, new_tensor_dir, new_midi_dir, model_path)`: reconstruct a directory of midi tensors using the VQ-VAE model; save as tensors and midis
* `save_graphs(midi_path, save_path)`: save pianoroll images of midis
* `show_result_graphs(yaml_name)`: show reconstruction error and perplexity graphs over model training
