import tensorflow as tf
import numpy as np
import pickle
import keras

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import print_summary
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D,Conv1D , Convolution1D , Activation
from keras.layers import MaxPooling2D, Dropout,MaxPooling1D,GlobalAveragePooling1D

from data_utils import loadFromPickle

from keras.utils import np_utils

def prepress_labels(labels):
    labels = np_utils.to_categorical(labels) # one-hot编码 把类别id转换为表示当前类别的向量，比如0 1 2 =》 [[1 0 0] [0 1 0] [0 0 1]]
    return labels


def keras_model_1dconv(input_shape,num_classes):
    model = Sequential()
    # model.add(keras.layers.core.Reshape((20, 64), input_shape=input_shape))
    model.add(Conv1D(100, 1, activation='relu', input_shape=input_shape))
    # model.add(Conv1D(100, 1, activation='relu'))
    model.add(MaxPooling1D(2))
    # model.add(Conv1D(160, 1, activation='relu'))
    model.add(Conv1D(160, 1, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    filepath = "1dconv.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model, callbacks_list

    # print(model.summary())   

def keras_model1(input_shape,num_of_classes):
    # 构建模型
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    filepath = "asr_mfcc_dense_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model, callbacks_list

def keras_model(input_shape,num_of_classes):
    num_of_classes = num_of_classes
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(140, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(70, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "asr_mfcc_conv2d_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list    

def test_model():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    features=features.reshape(features.shape[0],64,20,1)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    model, callbacks_list = keras_model((64,20,1,),len(labels[0]))
    print_summary(model)
    model.fit(train_x, train_y, batch_size=128, epochs=5, verbose=1, validation_data=(test_x, test_y),
    	callbacks=[TensorBoard(log_dir="TensorBoard")])

    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度

    model.save('asr_mfcc_conv2d_model.h5') # 保存训练模型

def test_model1():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    # features=features.reshape(features.shape[0],20,32,1)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    model, callbacks_list = keras_model1((20*64,),len(labels[0]))
    print_summary(model)
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=(test_x, test_y),
    	callbacks=[TensorBoard(log_dir="TensorBoard")])

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度
    model.save('asr_mfcc_dense_model.h5') # 保存训练模型

def test_1dconv():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    features=features.reshape(features.shape[0],32,20)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.2)
    model, callbacks_list = keras_model_1dconv((32,20),len(labels[0]))
    print_summary(model)
    model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1, validation_data=(test_x, test_y),
        callbacks=[TensorBoard(log_dir="TensorBoard")])

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度
    model.save('asr_model.h5') # 保存训练模型

if __name__ == '__main__':
	test_1dconv()