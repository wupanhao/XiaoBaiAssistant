import librosa
import numpy as np
import os
import cv2
import pickle
from keras.utils import np_utils

def get_mfcc(wav_path,samples=16000):
	y, sr = librosa.load(wav_path,sr=None)
	if len(y)<samples:
		y = np.concatenate((y, np.array([0]*(samples-len(y)))), axis=0)
	elif len(y) > samples:
		y = y[:samples]
	# print(y,sr)
	mfccs = librosa.feature.mfcc(y=y, sr=sr)
	return mfccs

def get_labels(data_dir):
    dirs = os.listdir(data_dir)
    dirs.sort()
    return dirs

def load_data(data_dir):
    MAX_NUM = 1000
    x_load = []
    y_load = []
    labels = get_labels(data_dir)
    dirs = labels
    for cat in dirs: #load directory
        files_dir = data_dir + cat 
        files = os.listdir(files_dir)
        for file in files[:MAX_NUM]:
            file_path = files_dir + "\\" + file
            mfccs = get_mfcc(file_path) # shape (20 , 32)
            x = np.array(mfccs).astype('float32')
            x_load.append(x)
            y_load.append(labels.index(cat))  # directory name as label
    return x_load,y_load

def dump_picle(features, labels):
    features = np.array(features).astype('float32')
    labels = np.array(labels).astype('float32')
    print(features.shape)
    print(labels.shape)
    features=features.reshape(features.shape[0],features.shape[1] * features.shape[2])
    print(features.shape)
    print(labels.shape)
    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)

def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels

def prepress_labels(labels):
    labels = np_utils.to_categorical(labels) # one-hot编码 把类别id转换为表示当前类别的向量，比如0 1 2 =》 [[1 0 0] [0 1 0] [0 0 1]]
    return labels

if __name__ == '__main__':
	f = "test.wav"
	mfccs = test2(f)
	print( mfccs[0] )
	print( mfccs.shape )