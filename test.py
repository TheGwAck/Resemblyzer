from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import sys
import numpy as np

path_1 = sys.argv[1]
path_2 = sys.argv[2]

fpath_1 = Path(path_1)
wav_1 = preprocess_wav(fpath_1)
fpath_2 = Path(path_2)
wav_2 = preprocess_wav(fpath_2)

encoder = VoiceEncoder()
embed_1 = encoder.embed_utterance(wav_1)
embed_2 = encoder.embed_utterance(wav_2)

np.set_printoptions(precision=3, suppress=True)
print(embed)