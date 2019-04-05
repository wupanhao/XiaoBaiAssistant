from flask import Flask
from flask import request,Response
from flask import render_template
from keras.models import load_model
import tensorflow as tf
import base64
import numpy as np
import cv2
import os
import json

from data_utils import get_labels

app = Flask(__name__)

graph = tf.get_default_graph()
model = load_model('QuickDraw_conv82.h5') # 加载训练模型
data_dir = ".\\npy_data\\"
label_names = get_labels(data_dir)

@app.route('/labels')
def get_labels():
	# return json.dumps(label_names)
	return Response(json.dumps(label_names), mimetype='application/json')

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
			pred_probab = keras_predict(model, gray)
			pred_class = list(pred_probab[0]).index(max(pred_probab[0]))
			print("may be " , label_names[pred_class] , " probab: " ,  pred_probab[0])
			formated = list( map(lambda x,i : (x.item(),label_names[i]) , pred_probab[0],[i for i in range(len(label_names))]) )
			return json.dumps(sorted(formated,reverse=True))
	else :
		# return "hello"
		return app.send_static_file('index.html')

def keras_predict(model, image):
    processed = keras_process_image(image)
    # print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)
    return pred_probab

def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

if __name__ == '__main__':  # pragma: no cover
    app.run(port=80)