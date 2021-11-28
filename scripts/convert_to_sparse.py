import scipy
import numpy as np
from pathlib import Path
import os
from scipy import sparse 
import pickle
from tqdm import tqdm

tensor_dir = Path('../data/all_midi_tensors') # all_midi_tensors_normed
sparse_dir = Path('../data/all_midi_tensors_sparse')

file_list = os.listdir(tensor_dir)

for f in tqdm(file_list): 
    cur_arr = np.load(tensor_dir / f)
    sparse_arr = scipy.sparse.csr_matrix(cur_arr)
    with open(sparse_dir / f, 'wb') as outfile:
        pickle.dump(sparse_arr, outfile)
print("DONE!")
