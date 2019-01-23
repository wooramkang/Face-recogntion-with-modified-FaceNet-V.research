from keras import backend as K
import time
from multiprocessing.dummy import Pool
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
from detect_landmarks_plus_affineTransform import *
import time

def id_generator():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    k = 0
    img_path = "id_pic/"
    folders = os.listdir(img_path)
    name_count = 0

    for name in folders:
        name_count = name_count + 1

    os.mkdir("id_pic/"+str(name_count)+"/")

    while vc.isOpened():

        now = int(time.time())
        _, frame = vc.read()
        cv2.waitKey(100)
        # _, _, frame = make_transformed_faceset(frame)
        '''
         written by wooramkang 2018.08. 23
    
         affine transform added
        '''
        #_, frame = make_transformed_faceset(frame)
        img = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) >= 2:
            continue
        else:
            for (x, y, w, h) in faces:

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                # part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                part_image = img[y1:y2, x1:x2]
                part_image = cv2.resize(part_image, (96, 96))
                cv2.imwrite('id_pic/' + str(name_count)+"/"+ str(k) + '.jpg', part_image)
                k = (k + 1) % 10000
                print(k)
                img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow("preview", img)

        key = cv2.waitKey(100)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")

if __name__ == "__main__":
    id_generator()