#!coding:utf-8
import os
from scipy.io import wavfile
import numpy as np
import librosa
import pyaudio
import wave
import cv2

from_dir = '.\data_raw\\zh\\'
to_dir = '.\data_gen\\'

# record command:
# arecord -d 1 -r 16000 -c 1 -t wav -f S16_LE record.wav

def write_wav2(file_name,sr,y):
    dir_name = file_name.split('_')[0]
    dest_dir = to_dir + dir_name +"/"    
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    file_path = dest_dir+file_name
    print(file_path)
    wavfile.write(file_path, sr, y)  # 写入音频
    return file_path

def write_wav(file_name,sr,y):
    maxv = np.iinfo(np.int16).max
    dir_name = file_name.split('_')[0]
    dest_dir = to_dir + dir_name +"/"    
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    file_path = dest_dir+file_name	
    wavfile.write(file_path, sr, (y*maxv).astype(np.int16))  # 写入音频
    # librosa.output.write_wav(file_path, y=(y*maxv).astype(np.int16), sr=sr)
    return	file_path
    '''
    with open(file_path,mode='w+b') as f:
        wav_fp = wave.open(f, 'wb')
        wav_fp.setnchannels(1)
        wav_fp.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wav_fp.setframerate(sr)
        #print(type(frames)) # list in Python3 , need bytes
        wav_fp.writeframes(b''.join(y))
        wav_fp.close()
        # f.seek(0)
        # return self.active_stt_engine.transcribe(f)
        return	file_path
	'''

def clip(wav_path,duration=2):
    y, sr = librosa.load(wav_path,sr = 16000)  # 读取音频
    print(y.shape, sr)
    samples = sr*duration
    if len(y) < samples : 
        y = np.concatenate((y, np.array([0]*(samples-len(y)))), axis=0)
    _ , file_name = os.path.split(wav_path)
    file_name = file_name.replace('.wav','_clip'+str(duration)+'s.wav')
    return write_wav(file_name, sr, y[:sr*duration])

def roll(wav_path,offset=100): #offset 100ms
    y, sr = librosa.load(wav_path, sr = 16000)  # 读取音频
    y = np.roll(y, int(-sr/1000*offset))
    # print y.shape, sr
    _ , file_name = os.path.split(wav_path)
    file_name = file_name.replace('.wav','_roll'+str(offset)+'ms.wav')
    return write_wav(file_name, sr, y)

def tune(wav_path,factor=1.2):
    y, sr = librosa.load(wav_path, sr = 16000)  # 读取音频
    ly = len(y)
    y_tune = cv2.resize(y, (1, int(len(y) * factor))).squeeze()
    lc = len(y_tune) - ly
    if lc > 0:
    	y_tune = y_tune[int(lc / 2):int(lc / 2) + ly]
    else :
    	n = int(abs(lc)/2)
    	new_tune = [0 for i in range(n)]
    	new_tune.extend(y_tune)
    	n = ly - len(new_tune)
    	new_tune.extend([0 for i in range(n)]) 
    	y_tune = np.array(new_tune).astype('float32')
    # print y.shape, sr
    _ , file_name = os.path.split(wav_path)
    file_name = file_name.replace('.wav','_tune'+str(factor)[:4]+'.wav')
    return write_wav(file_name, sr, y_tune)  # 写入音频

def tune_test(wav_path,factor=1.2):
    y, sr = librosa.load(wav_path, sr = 16000)  # 读取音频
    ly = len(y)
    y_tune = cv2.resize(y, (1, int(ly * factor))).squeeze()
    # y_tune = cv2.resize(y_tune, (1, ly)).squeeze()
    # lc = len(y_tune) - ly
    # y_tune = y_tune[int(lc / 2):int(lc / 2) + ly]
    # print y.shape, sr
    _ , file_name = os.path.split(wav_path)
    file_name = file_name.replace('.wav','_tune'+str(factor)[:4]+'.wav')
    return write_wav(file_name, sr, y_tune)  # 写入音频

def noise(wav_path,factor=0.02):
    y, sr = librosa.load(wav_path, sr = 16000)  # 读取音频
    wn = np.random.randn(len(y))
    y = np.where(y != 0.0, y + factor * wn, 0.0).astype(np.float32)  # 噪声不要添加到0上！
    # print y.shape, sr
    _ , file_name = os.path.split(wav_path)
    file_name = file_name.replace('.wav','_noise'+str(factor)[:6]+'.wav')
    return write_wav(file_name, sr, y)  # 写入音频

def speedx(sound_array, factor):
    """ 将音频速度乘以任意系数`factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def stretch(sound_array, f, window_size, h):
    """ 将音频按系数`f`拉伸 """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( int(len(sound_array) /f) + window_size).astype('complex128')

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):
        # print(i)
        i = int(i)
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
        result[i2 : i2 + window_size] += (hanning_window*a2_rephased)

    result = ((2**(16-4)) * result/result.max()) # 归一化 (16bit)

    return result.astype('int16')

def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ 将一段音频的音高提高``n``个半音 """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)

def gen50tones(wav_path,tones=range(-25,25)):
    _ , file_name = os.path.split(wav_path)	
    fps, raw_sound = wavfile.read(wav_path)
    transposed = [pitchshift(raw_sound, n) for n in tones]
    for i,new_sound in enumerate(transposed):
        file_name = file_name.replace('.wav','_tones'+str(i)+'.wav')
        write_wav2(file_name,fps,new_sound)

def load_files_test(data_dir):
	files = os.listdir(data_dir)
	print(files)
	for file in files[:1]:
		# gen50tones(data_dir+file,tones=range(-5,5))
		# break

		new_file = clip(data_dir+file)
		os.system('aplay '+ new_file)
		# new_file = roll(data_dir+file)
		# os.system('aplay '+ new_file)
		new_file = tune(data_dir+file,factor=1.2)
		os.system('aplay '+ new_file)
		new_file = tune(data_dir+file,factor=0.8)
		os.system('aplay '+ new_file)		
		# new_file = noise(data_dir+file)
		# os.system('aplay '+ new_file)

def load_files(data_dir):
    files = os.listdir(data_dir)
    print(files)
    for file in files:
        new_file = clip(data_dir+file)
        for i in range(1,15):
            new_file2 = roll(new_file,offset=i*5)
            for j in range(1,20):
                new_file3 = tune(new_file2,factor=0.75+j*0.04)
                for k in range(1,3):
                    noise(new_file3,factor = 0.004*k)

if __name__ == '__main__':
	load_files(from_dir)
