import numpy as np
import os
import cv2

from_dir = ".\\npy_data\\"
to_dir = ".\\img_data\\"
files = os.listdir(from_dir)

def write_image(cat,index,array):
    dest_dir = to_dir + cat +"\\"
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    cv2.imwrite(dest_dir + cat + "_" +str(index) + ".png", array)

def load_data():
    count = 0
    for file in files:
        # print(os.path.splitext(file))
        catname_full,_ = os.path.splitext(file)
        catname = catname_full.split('_')[-1]
        # print(catname)
        file = from_dir + file
        cat = os.path.basename(file)
        # print(cat)
        imgs = np.load(file)
        print(imgs.shape)
        # print(np.shape(imgs)) # (133572, 784)

        # imgs = imgs.astype('float32') / 255.
        imgs = imgs[0:100, :]

        for index,img in enumerate(imgs):
            img = img.reshape(28,28)
            write_image(catname,index,img)

load_data()
