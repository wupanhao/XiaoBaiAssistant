import librosa
import numpy as np
import os
import pickle

from scipy.io import wavfile

# record command:
# arecord -d 1 -r 16000 -c 1 -t wav -f S16_LE record.wav

def clip(wav_path,duration=1):
    y, sr = librosa.load(wav_path)  # 读取音频
    # print y.shape, sr
    samples = sr*duration
    if len(y) < samples : 
        y = np.concatenate((y, np.array([0]*(samples-len(y)))), axis=0)
    file_path = wav_path.reaplce('.wav','_clip'+str(duration)+'s.wav')
    wavfile.write(file_path, sr, y[:sr*duration])  # 写入音频

def roll(wav_path,offset=100): #offset 100ms
    y, sr = librosa.load(wav_path)  # 读取音频
    y = np.roll(y, sr/1000*samples)
    # print y.shape, sr
    file_path = wav_path.reaplce('.wav','_roll'+str(offset)+'ms.wav')
    wavfile.write(file_path, sr, y)  # 写入音频
    return file_path

def tune(wav_path,factor=1.2):
    y, sr = librosa.load(wav_path)  # 读取音频
    ly = len(y)
    y_tune = cv2.resize(y, (1, int(len(y) * factor))).squeeze()
    lc = len(y_tune) - ly
    y_tune = y_tune[int(lc / 2):int(lc / 2) + ly]
    # print y.shape, sr
    file_path = wav_path.reaplce('.wav','_tune'+str(factor)[:4]+'.wav')
    wavfile.write(file_path, sr, y_tune)  # 写入音频
    return file_path

def noise(wav_path,factor=0.02):
    y, sr = librosa.load(wav_path)  # 读取音频
    wn = np.random.randn(len(y))
    y = np.where(y != 0.0, y + factor * wn, 0.0)  # 噪声不要添加到0上！
    # print y.shape, sr
    file_path = wav_path.reaplce('.wav','_noise'+str(factor)[:6]+'.wav')
    wavfile.write(file_path, sr, y)  # 写入音频

def speedx(sound_array, factor):
    """ 将音频速度乘以任意系数`factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def stretch(sound_array, f, window_size, h):
    """ 将音频按系数`f`拉伸 """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( len(sound_array) /f + window_size)

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # 两个可能互相重叠的子数列
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # 按第一个数列重新同步第二个数列
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # 加入到结果中
        i2 = int(i/f)
        result[i2 : i2 + window_size] += hanning_window*a2_rephased

    result = ((2**(16-4)) * result/result.max()) # 归一化 (16bit)

    return result.astype('int16')

def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ 将一段音频的音高提高``n``个半音 """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)

def gen50tones(wav_path,tones=range(-25,25))
    fps, bowl_sound = wavfile.read(wav_path)
    transposed = [pitchshift(bowl_sound, n) for n in tones]

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
    data_dir = '.\\data\\'
    dirs = get_labels(data_dir)
    dump_label_name(dirs)
    ddd = load_label_name()
    print(ddd)
    '''
	f = "test.wav"
	mfccs = test2(f)
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