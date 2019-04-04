import wave
import numpy as np
import os
from keras.models import load_model
import librosa


def get_wav_mfcc(wav_path):
    y, sr = librosa.load(wav_path,sr=None)
    if len(y)<16000:
        y = np.concatenate(( np.array([0]*(16000-len(y))), y), axis=0)
        # y = np.concatenate(( y, np.array([0]*(16000-len(y)))), axis=0)
    elif len(y) > 16000:
        y = y[:16000]
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.reshape(mfccs,(-1,640))[0]

def test_wav_mfcc(wav_path):
    y, sr = librosa.load(wav_path,sr=None)
    print(y)
    # y = np.concatenate(( np.array([0]*(32000-len(y))), y), axis=0)
    y = y[6000:10000]
    if len(y)<16000:
        y = np.concatenate(( np.array([0]*(16000-len(y))), y), axis=0)
        # y = np.concatenate(( y, np.array([0]*(16000-len(y)))), axis=0)
    elif len(y) > 16000:
        y = y[:16000]
    print(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.reshape(mfccs,(-1,640))[0]

if __name__ == '__main__':
    # 构建模型
    model = load_model('asr_model_weights.h5') # 加载训练模型
    wavs=[]
    # wavs.append(get_wav_mfcc(".\\data\\test\\seven\\00970ce1_nohash_0.wav"))wrong answer!
    # wavs.append(get_wav_mfcc(".\seven (2).wav"))
    wavs.append(test_wav_mfcc(".\\data\\test\\seven\\890cc926_nohash_3.wav"))
    # wavs.append(test_wav_mfcc(".\stop.wav"))
    X=np.array(wavs)
    print(X.shape)
    result=model.predict(X[0:1])[0] # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
    print("识别结果",result[0],result[1])
    #  因为在训练的时候，标签集的名字 为：  0：seven   1：stop    0 和 1 是下标
    name = ["seven","stop"] # 创建一个跟训练时一样的标签集
    ind=0 # 结果中最大的一个数
    for i in range(len(result)):
        if result[i] > result[ind]:
            ind=1
    print("识别的语音结果是：",name[ind])




