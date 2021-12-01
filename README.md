# Midi Generation
Research project started summer 2021. 
### Goal: generate cohesive midis possessing the structure and repetition found in most modern music. 

### Dependencies
* python
* pytorch
* numpy
* pretty_midi
* pygame
* tqdm
* yaml

Install all with: `conda install pygame tqdm yaml pretty_midi numpy pytorch`
Set up your virtual environment with python 3.8.12

### Usage:

To train the VQ-VAE model:
1. Clone repository to your local machine
2. Navigate to /scripts directory
3. Run `python test_vqvae.py` with the following (optional) arguments:(
    * `-d`: path to data directory (midi tensors)
    * `-m`: path where you want model saved
    * `-o`: path to desired results directory
    * `-l`: specify loss function (default is MSE, using flag switches to MAE)
    * `-v`: increase logging verbosity