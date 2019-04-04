import librosa
import wave
import numpy as np

def test(wav_path):
    wavs=[]
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int    
    # wavs.append(get_wav_mfcc(".\\data\\test\\seven\\00970ce1_nohash_0.wav"))
def test2(wav_path):
	y, sr = librosa.load(wav_path,sr=None)
	if len(y)<16000:
		y = np.concatenate((y, np.array([0]*(16000-len(y)))), axis=0)
		# y.concatenate( np.array([0]*(16000-len(y))), axis=0)
	elif len(y) > 16000:
		y = y[:16000]
	print(y,sr)
	mfccs = librosa.feature.mfcc(y=y, sr=sr)
	return mfccs

if __name__ == '__main__':
	f = ".\\data\\test\\seven\\00970ce1_nohash_0.wav"
	# f = ".\stop.wav"
	test(f)
	mfccs = test2(f)
	print( mfccs[0] )
	print( mfccs.shape )