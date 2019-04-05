import cv2
from keras.models import load_model
import numpy as np
import os
from data_utils import get_labels,loadFromPickle,prepress_labels
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import print_summary

def validate():
    data_dir = ".\\npy_data\\"
    label_names = get_labels(data_dir)
    model = load_model('QuickDraw_conv82.h5') # 加载训练模型

    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    features=features.reshape(features.shape[0],28,28,1)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    print_summary(model)

    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度	

if __name__ == '__main__':
	validate()