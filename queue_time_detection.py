import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from collections import deque

############################################
directory = r'data\waiting_line.mp4'
bounding_box_topleft = (250,3)
bounding_box_bottomright = (630,351)
frame_rate = 1
n_frame = 5
service_rate = 3 # customer per minute
############################################

#loading model
model = tf.keras.models.load_model('final_model3.h5')
# model.summary()

images = []
capture = cv2.VideoCapture(directory)
avg_results = deque(maxlen = n_frame)
while(capture.isOpened()):

    success , frame = capture.read()
    
    if not success:
        break
        
    size = frame.shape
    cv2.rectangle(frame, bounding_box_topleft, bounding_box_bottomright, (255, 0 , 255), 1)
    crop_frame = frame[bounding_box_topleft[1]:bounding_box_bottomright[1],bounding_box_topleft[0]:bounding_box_bottomright[0]]
    frame_gray = cv2.cvtColor(crop_frame,cv2.COLOR_BGR2GRAY)
    results = model.predict(np.expand_dims(frame_gray, axis = 0))
    n_people = np.round(np.sum(results))
    avg_results.append(n_people)
    people = np.round(np.mean(avg_results))

    cv2.putText(frame,f'No. of people in Queue:{people}',
                  (104, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    cv2.putText(frame,f'service rate:{service_rate}',
                  (104, 34+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    cv2.putText(frame,f'Queue waiting time:{np.round(people/service_rate)} min',
                  (104, 34+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    
    # cv2.imshow("waiting_line",frame)
    images.append(frame)
    if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()


# saving video
out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (size[1],size[0]))
for i in range(len(images)):
    out.write(images[i])
out.release()




    