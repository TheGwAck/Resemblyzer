from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys


# The demos are ordered so as to make the explanations in the comments consistent. If you only 
# care about running the code, then you don't have to follow that order.
# DEMO 01: we'll show how to compare speech segments (=utterances) between them to get a metric  
# on how similar their voices sound. We expect utterances from the same speaker to have a high 
# similarity, and those from distinct speakers to have a lower one. 


# The neural network will automatically use CUDA if it'speaker available on your machine, otherwise it 
# will use the CPU. You can enforce a device of your choice by passing its name as argument to the 
# constructor. The model might take a few seconds to load with CUDA, but it then executes very 
# quickly.
encoder = VoiceEncoder()

# We'll use a smaller version of the dataset LibriSpeech test-other to run our examples. This 
# smaller dataset contains 10 speakers with 10 utterances each. N.B. "wav" in variable names stands
# for "waveform" and not the wav file extension.
# wav_fpaths = list(Path("audio_data", "librispeech_test-other").glob("**/*.flac"))
# Group the wavs per speaker and load them using the preprocessing function provided with 
# resemblyzer to load wavs in memory. It normalizes the volume, trims long silences and resamples 
# the wav to the correct sampling rate.

path = '/content/drive/MyDrive/Collabera_Jawad/myaudio'

speaker_1 = sys.argv[1]
speaker_2 = sys.argv[2]


wav_1_fpaths = list(Path(path, speaker_1).glob("*.wav"))
wav_2_fpaths = list(Path(path, speaker_2).glob("*.wav"))


speaker_1_wavs = {speaker_1: list(map(preprocess_wav, wav_1_fpaths)) for wav_1_fpaths in
                groupby(tqdm(wav_1_fpaths, "Preprocessing wavs", len(wav_1_fpaths), unit="wavs"), 
                        lambda wav_1_fpath: wav_1_fpath.parent.stem)}
speaker_2_wavs = {speaker_2: list(map(preprocess_wav, wav_2_fpaths)) for wav_2_fpaths in
                groupby(tqdm(wav_2_fpaths, "Preprocessing wavs", len(wav_2_fpaths), unit="wavs"), 
                        lambda wav_2_fpath: wav_2_fpath.parent.stem)}
speaker_1_wavs.update(speaker_2_wavs) 
speaker_wavs = speaker_1_wavs


## Similarity between two utterances from each speaker
# Embed two utterances A and B for each speaker
embeds_a = np.array([encoder.embed_utterance(wavs[0]) for wavs in speaker_wavs.values()])
embeds_b = np.array([encoder.embed_utterance(wavs[1]) for wavs in speaker_wavs.values()])
# Each array is of shape (num_speakers, embed_size) which should be (10, 256) if you haven't 
# changed anything.
print("Shape of embeddings: %s" % str(embeds_a.shape))

# Compute the similarity matrix. The similarity of two embeddings is simply their dot 
# product, because the similarity metric is the cosine similarity and the embeddings are 
# already L2-normed.
# Short version:
utt_sim_matrix = np.inner(embeds_a, embeds_b)
# Long, detailed version:
utt_sim_matrix2 = np.zeros((len(embeds_a), len(embeds_b)))
for i in range(len(embeds_a)):
    for j in range(len(embeds_b)):
        # The @ notation is exactly equivalent to np.dot(embeds_a[i], embeds_b[i])
        utt_sim_matrix2[i, j] = embeds_a[i] @ embeds_b[j]
assert np.allclose(utt_sim_matrix, utt_sim_matrix2)


## Similarity between two speaker embeddings
# Divide the utterances of each speaker in groups of identical size and embed each group as a
# speaker embedding
spk_embeds_a = np.array([encoder.embed_speaker(wavs[:len(wavs) // 2]) \
                         for wavs in speaker_wavs.values()])
spk_embeds_b = np.array([encoder.embed_speaker(wavs[len(wavs) // 2:]) \
                         for wavs in speaker_wavs.values()])
spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)


## Draw the plots
fix, axs = plt.subplots(2, 2, figsize=(8, 10))
labels_a = ["%s-A" % i for i in speaker_wavs.keys()]
labels_b = ["%s-B" % i for i in speaker_wavs.keys()]
mask = np.eye(len(utt_sim_matrix), dtype=np.bool)
plot_similarity_matrix(utt_sim_matrix, labels_a, labels_b, axs[0, 0],
                       "Cross-similarity between utterances\n(speaker_id-utterance_group)")
plot_histograms((utt_sim_matrix[mask], utt_sim_matrix[np.logical_not(mask)]), axs[0, 1],
                ["Same speaker", "Different speakers"], 
                "Normalized histogram of similarity\nvalues between utterances")
plot_similarity_matrix(spk_sim_matrix, labels_a, labels_b, axs[1, 0],
                       "Cross-similarity between speakers\n(speaker_id-utterances_group)")
plot_histograms((spk_sim_matrix[mask], spk_sim_matrix[np.logical_not(mask)]), axs[1, 1],
                ["Same speaker", "Different speakers"], 
                "Normalized histogram of similarity\nvalues between speakers")
plt.show()
