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
save_dir = "saved/"
bounding_box_topleft = (7, 74)
bounding_box_bottomright = (612,408)
filename = "saved.jpg"
threshold = 0.2
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
    cv2.rectangle(img, bounding_box_topleft, bounding_box_bottomright, (255, 0 , 255), 1)
    crop_img = img[bounding_box_topleft[1]:bounding_box_bottomright[1],bounding_box_topleft[0]:bounding_box_bottomright[0]]

    results = tfnet.return_predict(crop_img) #returns list of dictionaries
    n_people = 0
    for dic  in results:
        if dic["label"] == "person" and dic["confidence"] >= threshold:
            bbox_dic = dic['topleft']
            bbox_topleft = tuple(bbox_dic.values())
            bbox_dic = dic['bottomright']
            bbox_bottomright = tuple(bbox_dic.values())
            cv2.rectangle(crop_img, bbox_topleft, bbox_bottomright, (0, 255 , 0), 1)
            n_people += 1

    cv2.putText(crop_img,f'No. of people in Queue:{n_people}',
                  (315, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.imshow("Queue",crop_img)
    cv2.imwrite(os.path.join(save_dir , 'test.jpg'),crop_img)

    print(n_people)
    cv2.waitKey()