# thanks to https://github.com/akshaybahadur21/QuickDraw

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense,Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
import keras
from keras.callbacks import TensorBoard

from data_utils import loadFromPickle,prepress_labels,get_labels,MAX_NUM

def keras_model1(input_shape,num_classes):
    # 构建模型
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    filepath = "model1.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model, callbacks_list

def keras_model(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model, callbacks_list

def test_model():
    features, labels = loadFromPickle()
    labels_count = int(len(labels)/MAX_NUM)
    # features, labels = augmentData(features, labels)
    features, labels = shuffle(features, labels)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    model, callbacks_list = keras_model( (28,28,1,) , labels_count ) 
    print_summary(model)
    # model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=2, batch_size=64,
    #          callbacks=[TensorBoard(log_dir="TensorBoard")])
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=2, batch_size=64)
    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度       
    model.save('model.h5')

def test_model1():
    features, labels = loadFromPickle()
    labels_count = int(len(labels)/MAX_NUM)
    # features, labels = augmentData(features, labels)
    features, labels = shuffle(features, labels)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    # train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    # test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    model, callbacks_list = keras_model1( (28*28,) , labels_count ) 
    print_summary(model)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=500, batch_size=256)
    # model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=500, batch_size=256,
    #          callbacks=[TensorBoard(log_dir="TensorBoard")])

    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度    

    model.save('model1.h5')

test_model1()

