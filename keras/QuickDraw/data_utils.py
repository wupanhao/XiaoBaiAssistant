import librosa
import numpy as np
import os
import cv2
import pickle
from keras.utils import np_utils

def get_labels(data_dir):
    labels = []
    files = os.listdir(data_dir)
    files.sort()
    for file in files:
      catname_full,_ = os.path.splitext(file)
      catname = catname_full.split('_')[-1]
      labels.append(catname)
    print(labels)
    return labels

def load_data(data_dir): #From Directory
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
    features=features.reshape(features.shape[0]*features.shape[1], features.shape[2])
    labels=labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2])
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

def write_image(cat,index,array):
    dest_dir = to_dir + cat +"\\"
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    cv2.imwrite(dest_dir + cat + "_" +str(index) + ".png", array)

def load_npy_data(data_dir):
    MAX_NUM = 1000
    x_load = []
    y_load = []
    labels = get_labels(data_dir)
    dirs = labels
    write_img_file = False
    files = os.listdir(data_dir)
    for file in files:
        catname_full,_ = os.path.splitext(file)
        catname = catname_full.split('_')[-1]
        file = data_dir + file
        cat = os.path.basename(file)
        imgs = np.load(file)
        print(imgs.shape)
        # print(np.shape(imgs)) # (133572, 784)
        # imgs = imgs.astype('float32') / 255.
        imgs = imgs[:MAX_NUM, :]
        x_load.append(imgs)
        y = [labels.index(catname) for _ in range(MAX_NUM)]
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)
        if write_img_file:
            for index,img in enumerate(imgs):
                img = img.reshape(28,28)
                write_image(catname,index,img)
    return x_load,y_load
if __name__ == '__main__':
    from_dir = ".\\npy_data\\"
    to_dir = ".\\img_data\\"
    features, labels = load_npy_data(from_dir)    
    dump_picle(features, labels)
