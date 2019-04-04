import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os

img_dir = ".\\img_data\\"
dirs = os.listdir(img_dir)
# labels = map(lambda x: os.path.splitext(x)[0].split('_')[-1],dirs)
labels = []
for label in dirs:
    label = os.path.splitext(label)[0].split('_')[-1]
    labels.append(label)
print(labels)

model = load_model('QuickDraw.h5')

def test_data():
    count = 0.0
    right = 0.0
    for dir in dirs:
        catname_full,_ = os.path.splitext(dir)
        catname = catname_full.split('_')[-1]
        # print(catname)

        for index,file in enumerate(os.listdir(img_dir+dir)):
            count = count + 1
            img = cv2.imread(img_dir+dir+"\\"+file)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            print(gray.shape)
            pred_probab, pred_class = keras_predict(model, gray)
            print("should be " , catname , " , result is " , labels[pred_class], " probab: " ,  pred_probab)
            right = right + ( catname == labels[pred_class])
    print("Accuracy: ",right/count) 

def keras_predict(model, image):
    processed = keras_process_image(image)
    # print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

if __name__ == '__main__':
    # i = cv2.imread("test.png")
    # print(i)
    # pred_probab, pred_class = keras_predict(model, i)
    # print("should be " , "bird" , " , result is " , labels[pred_class], " probab: " ,  pred_probab) 
    # pass
    test_data()
