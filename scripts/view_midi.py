from musicautobot.numpy_encode import *
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.multitask_transformer import *
from musicautobot.utils import midifile
from music21 import *
import sys

n = note.Note("D#3")
n.duration.type = 'half'
n.show()

#midi_file = Path(DATA_DIR + 'new_data\\ableton_midi.mid')
#vocab = MusicVocab.create()
#item = MusicItem.from_file(midi_file, vocab); item.show()