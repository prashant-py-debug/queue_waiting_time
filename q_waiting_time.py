import warnings
warnings.filterwarnings('ignore')
from darkflow.net.build import TFNet # Importing and buolding the yolo model on predefined yolo weights
import numpy as np
import cv2
import os
#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf

############################################
directory = "data/"
bounding_box_topleft = (7, 74)
bounding_box_bottomright = (612,408)
filename = "saved.jpg"

############################################

# Config TF, set True if using GPU
config = tf.ConfigProto(log_device_placement = True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    options = {
            'model': './cfg/yolo.cfg',
            'load': './bin/yolo.weights',
            'threshold': 0.3,
            # 'gpu': 1.0 # uncomment these if using GPU
               }
    tfnet = TFNet(options)


def image_reader(path):
    """
    fuction takes the string address as input
    outputs the list of image address
    """

    paths = os.listdir(path)
    images =[]
    for p in paths:
        images.append(os.path.join(path,p))
    return images


img_dir = image_reader(directory)
for i in img_dir:
    img = cv2.imread(i)
    cv2.rectangle(img, bounding_box_topleft, bounding_box_bottomright, (255, 0 , 255), 2)
    crop_img = img[74:407,7:612]

    results = tfnet.return_predict(crop_img) #returns list of dictionaries
    n_people = len(results) 

    cv2.putText(img,f'No. of people in Queue:{n_people}',
                  (300, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.imshow("Queue",img)
    cv2.imwrite(directory+filename,img)

    print(results)
    print(n_people)
    cv2.waitKey()