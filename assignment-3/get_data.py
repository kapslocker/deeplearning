import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_mfcc(inp):
    plt.figure(figsize = (10, 4))
    librosa.display.specshow(inp, x_axis = 'time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def getlabels(data_dir):
    ''' Generates pairs of numpy arrays (audio_seq, audio_seq_label).
        Label is decided as 1 for a boundary, 0 for not a boundary position.
        dir: TRAIN/TEST with tree view as:
                |-DRx
                |--speaker_id
                |---sentence.PHN
                |---sentence.WAV
                |---sentence.WRD '''
    for dialect in os.listdir(data_dir):
        for speaker in os.listdir(data_dir + '/' + dialect):
            files = os.listdir(data_dir + '/' + dialect + '/' + speaker)
            wav_files = [f for f in files if ".WAV" in f]
            phn_files = [p for p in files if ".PHN" in p]
            wrd_files = [w for w in files if ".WRD" in w]
            for (wav, wrd) in zip(wav_files, wrd_files):
                seq, sr = librosa.load(data_dir + '/' + dialect + '/' + speaker + '/' + wav)
                wrd_file = data_dir + '/' + dialect + '/' + speaker + '/' + wrd
                boundaries = list()
                with open(wrd_file) as mapping_file:
                    for line in mapping_file:
                        s = line.split(' ')
                        a = int(s[0])
                        b = int(s[1])
                        if a not in boundaries:
                            boundaries.append(a)
                        if b not in boundaries:
                            boundaries.append(b)
                seq_labels = [0] * len(seq)
                #----------------blah-----blah--blaaaaahhh--------
                for idx in xrange(boundaries[0]):
                    seq_labels[idx] = 1
                for idx in xrange(boundaries[-1], len(seq)):
                    seq_labels[idx] = 1
                for bdr in xrange(1, len(boundaries) - 1):
                    idx = boundaries[bdr]
                    for i in xrange(max(0,idx - 10), min(len(seq), idx + 10)):
                        seq_labels[i] = 1
                yield (seq, np.array(seq_labels))

# mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40)
#
# mfccs_delta = librosa.feature.delta(mfccs)
# mfccs_delta2 = librosa.feature.delta(mfccs, order = 2)
#
# print mfccs.shape
