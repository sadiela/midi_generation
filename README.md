# Midi Generation
Research project started summer 2021. 
### Goal: generate cohesive midis possessing the structure and repetition found in most modern music. 

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


### Usage:

#### `convert_to_sparse.py`
Takes a directory of tensor midi representations and converts them to sparse format. 

#### `midi_preprocessing.py`


#### `train_vqvae.py`

To train the VQ-VAE model:
1. Clone repository to your local machine
2. Navigate to /scripts directory
3. Run `python test_vqvae.py` with the following (optional) arguments:(
    * `-d`: path to data directory (midi tensors)
    * `-m`: path where you want model saved
    * `-o`: path to desired results directory
    * `-r`: result and model filestub
    * `-n`: whether or not to normalize the tensors (default is not to normalize)
    * `-b`: number of songs in a batch
    * `-l`: specify loss function (default is MSE, using flag switches to MAE)
    * `-v`: increase logging verbosity
    * `-q`: whether or not to quantize representations