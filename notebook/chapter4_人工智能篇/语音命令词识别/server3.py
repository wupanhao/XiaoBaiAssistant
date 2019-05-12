#!coding:utf-8
import sys
import os
from keras.models import load_model
from data_utils import load_label_name,get_mfcc,get_pcm
import numpy as np

root_dir = "/xiaobai/"
sys.path.append(root_dir)
from xiaobai import XiaoBai,BaseSkill

# model = load_model('asr_model.h5') # 加载训练模型
# model = load_model('asr_mfcc_conv1d_model.h5') # 加载训练模型
model = load_model('test_1dconv.h5') # 加载训练模型
# model = load_model('asr_mfcc_dense_model.h5') # 加载训练模型
label_names = load_label_name()
def predict(model):
    X=get_mfcc('record.wav',samples=16000)
    # X=get_pcm('record.wav',samples=32000)
    print(X.shape)
    #test dense model
    # X = np.reshape(np.array(X),(-1,32000))

    # test conv1d model
    X = np.reshape(np.array(X),(-1,32,20))
    # test conv2d model
    # X = np.reshape(np.array(X),(-1,64,40))    
    pred_probab = model.predict(X)
    pred_class = list(pred_probab[0]).index(max(pred_probab[0]))
    print("may be " , label_names[pred_class] , " probab: " ,  pred_probab[0][pred_class])
    formated = list( map(lambda x,i : (x.item(),label_names[i]) , pred_probab[0],[i for i in range(len(label_names))]) )
    lists = sorted(formated,reverse=True)
    top5 = lists[:5]
    print("top5 :",top5)
    return lists
def callback():
  os.system("arecord -d %d -r 16000 -c 1 -t wav -f S16_LE record.wav" % (1,) )   
  predict(model)
def main():
  keyword_model = root_dir+'resources/小白.pmdl'
  xiaobai = XiaoBai(keyword_model=keyword_model,callback=callback)
  xiaobai.listen_for_keyword()

if __name__ == '__main__':
    main()
