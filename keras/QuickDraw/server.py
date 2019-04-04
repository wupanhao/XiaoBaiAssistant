from flask import Flask
from flask import request
from flask import render_template
from keras.models import load_model
import tensorflow as tf
import base64
import numpy as np
import cv2
import os

from QuickDrawTest import *

app = Flask(__name__)

graph = tf.get_default_graph()

@app.route('/',methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		# print(request.form)
		img_b64encode = request.form.get("base64img","")
		# img_b64encode = "data:image/jpeg;base64,/9j/4AAQS......."
		img_b64decode = base64.b64decode(img_b64encode[23:])  # base64解码
		img_array = np.fromstring(img_b64decode,np.uint8) # 转换np序列
		# img_array = np.fromstring(img_b64decode,np.float32) # 转换np序列
		print(img_array.shape)
		img = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)  # 转换Opencv格式
		img = cv2.resize(img, (28, 28))
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		print(gray.shape)
		with graph.as_default():
			pred_probab, pred_class = keras_predict(model, gray)
			print("may be " , labels[pred_class] , " probab: " ,  pred_probab)
			return labels[pred_class]
	else :
		# return "hello"
		return app.send_static_file('index.html')

if __name__ == '__main__':  # pragma: no cover
    app.run(port=80)