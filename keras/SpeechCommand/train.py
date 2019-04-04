import tensorflow as tf
import numpy as np
import pickle
import keras

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import print_summary
from keras.layers import Dense
from keras.models import Sequential

from data_utils import loadFromPickle,prepress_labels

def keras_model(input_shape,output_shape):
    # 构建模型
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    filepath = "asr_mfcc_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model, callbacks_list


def main():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    model, callbacks_list = keras_model((640,),len(labels[0]))
    print_summary(model)
    model.fit(train_x, train_y, batch_size=128, epochs=100, verbose=1, validation_data=(test_x, test_y),
    	callbacks=[TensorBoard(log_dir="TensorBoard")])

    # 开始评估模型效果 # verbose=0为不输出日志信息
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 准确度

    model.save('asr_mfcc_model.h5') # 保存训练模型

if __name__ == '__main__':
	main()