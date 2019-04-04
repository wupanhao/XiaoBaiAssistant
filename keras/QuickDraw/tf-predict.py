#!/usr/bin/env python
 
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.python.platform import gfile
 
img_dir = ".\\img_data\\"
dirs = os.listdir(img_dir)
# labels = map(lambda x: os.path.splitext(x)[0].split('_')[-1],dirs)
labels = []
for label in dirs:
    label = os.path.splitext(label)[0].split('_')[-1]
    labels.append(label)
print(labels)
 
class_names = labels
img = cv2.imread("test.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
test_images = np.reshape(gray, (-1, 28, 28, 1))
# print(i)
# print("test_images.shape = {}".format(test_images.shape))
 
# Initialize a tensorflow session
with tf.Session() as sess:
    # Load the protobuf graph
    with gfile.FastGFile("QuickDraw2.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Add the graph to the session
        tf.import_graph_def(graph_def, name='')
 
    # Get graph
    graph = tf.get_default_graph()
 
    # Get tensor from graph
    pred = graph.get_tensor_by_name("dense_3/Softmax:0")
    
    # Run the session, evaluating our "c" operation from the graph
    res = sess.run(pred, feed_dict={'conv2d_1_input:0': test_images})
 
    # Print test accuracy
    pred_index = np.argmax(res[0])
    print(res)
    # Print test accuracy
    print('Predict:', pred_index, ' Label:', class_names[pred_index], 'GT:', "bird")