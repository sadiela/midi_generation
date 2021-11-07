import re  
import os
import tqdm
from pathlib import Path

# General utility functions (mostly for file management)
def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            print("file exists")
            counter += 1
        else:  # No match found
            print("no file")
            if suffix=='.p':
                print("will create pickle file")
            elif suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def remove_special_chars(dir):
    # removes special characters from all file names in the given directory
    for f in tqdm(os.listdir(dir)):
        r = re.sub(r'[^A-Za-z0-9_. ]', r'', f)
        if( r != f): # if name is different 
            if os.path.isfile(dir + f):
                os.replace(dir + f, dir + r)
            else: 
                os.rename(dir + f, dir + r)