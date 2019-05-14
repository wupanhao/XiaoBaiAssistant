import librosa
import numpy as np
import os
import pickle

def get_mfcc(wav_path,samples=32000):
    y, sr = librosa.load(wav_path,sr=None)
    if len(y)<samples:
        y = np.concatenate((y, np.array([0]*(samples-len(y)))), axis=0)
    elif len(y) > samples:
        y = y[:samples]
    # print(y,sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    # mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40)
    return np.array(mfccs).T

def get_pcm(wav_path,samples=32000):
    y, sr = librosa.load(wav_path,sr=None)
    if len(y)<samples:
        y = np.concatenate((y, np.array([0]*(samples-len(y)))), axis=0)
    elif len(y) > samples:
        y = y[:samples]
    # print(y,sr)
    # mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40)
    return y

def get_labels(data_dir):
    dirs = os.listdir(data_dir)
    dirs.sort()
    return dirs

def load_data(data_dir,samples=32000):
    MAX_NUM = 1000
    x_load = []
    y_load = []
    labels = get_labels(data_dir)
    dirs = labels
    for cat in dirs: #load directory
        files_dir = os.path.join(data_dir, cat) 
        files = os.listdir(files_dir)
        for file in files[:MAX_NUM]:
            file_path = os.path.join(files_dir, file)
            mfccs = get_mfcc(file_path,samples) # shape (20 , 32)
            x = np.array(mfccs).astype('float32')
            x_load.append(x)
            y_load.append(labels.index(cat))  # directory name as label
        print( files_dir + 'loaded ' )
    return x_load,y_load

def load_data_pcm(data_dir,samples=32000):
    MAX_NUM = 1000
    x_load = []
    y_load = []
    labels = get_labels(data_dir)
    dirs = labels
    for cat in dirs: #load directory
        files_dir = os.path.join(data_dir, cat) 
        files = os.listdir(files_dir)
        for file in files[:MAX_NUM]:
            file_path = os.path.join(files_dir, file)
            pcm = get_pcm(file_path,samples) # shape (32000,)
            x = np.array(pcm).astype('float32')
            x_load.append(x)
            y_load.append(labels.index(cat))  # directory name as label
        print( files_dir + 'loaded ' )
    return x_load,y_load

def dump_label_name(dirs):
    with open("label_names", "wb") as f:
        pickle.dump(dirs, f, protocol=4)    
def load_label_name():
    with open("label_names", "rb") as f:
        dirs = np.array(pickle.load(f))
    return dirs

def dump_picle(features, labels):
    features = np.array(features).astype('float32')
    labels = np.array(labels).astype('float32')
    print(features.shape)
    print(labels.shape)
    if len(features.shape) == 3 :
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

if __name__ == '__main__':
    data_dir = 'data'
    dirs = get_labels(data_dir)
    dump_label_name(dirs)
    ddd = load_label_name()
    print(ddd)
    '''
    f = "test.wav"
    mfccs = get_mfcc(f,32500)
    print( mfccs[0] )
    print( mfccs.shape )
    '''
'''
#pianopy
import pygame
 
pygame.mixer.init(fps, -16, 1, 512) # 太灵活了 ;)
screen = pygame.display.set_mode((640,480)) # 设置焦点
 
# 得到键盘的键位的正确顺序的列表
# ``keys`` 如 ['Q','W','E','R' ...] 一样排列
keys = open('typewriter.kb').read().split('\n')
 
sounds = map(pygame.sndarray.make_sound, transposed)
key_sound = dict( zip(keys, sounds) )
is_playing = {k: False for k in keys}
 
while True:
 
    event =  pygame.event.wait()
 
    if event.type in (pygame.KEYDOWN, pygame.KEYUP):
        key = pygame.key.name(event.key)
 
    if event.type == pygame.KEYDOWN:
 
        if (key in key_sound.keys()) and (not is_playing[key]):
            key_sound[key].play(fade_ms=50)
            is_playing[key] = True
 
        elif event.key == pygame.K_ESCAPE:
            pygame.quit()
            raise KeyboardInterrupt
 
    elif event.type == pygame.KEYUP and key in key_sound.keys():
 
        key_sound[key].fadeout(50) # 停止播放并50ms淡出
        is_playing[key] = False
'''