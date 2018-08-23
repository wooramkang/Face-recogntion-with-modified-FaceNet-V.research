
import cv2
import dlib
import numpy as np


def shape_to_np(shape, dtype):
    # create an empty numpy array
    coord = np.zeros(shape=(68, 2), dtype=dtype)

    # fill the numpy array with coordinates
    for i in range(0, 68):
        coord[i] = (shape.part(i). x, shape.part(i).y)

    # return the numpy array
    return coord


def affine_transform(shape_face, frame):
    shape_img = frame.shape

    rows = shape_img[0]
    cols = shape_img[1]

    w = float(shape_face[46][1]) - float(shape_face[37][1])
    w = float(w / (float(shape_face[46][0]) - float(shape_face[37][0])))

    angle = int(cv2.fastAtan2(w, 1))

    affine = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    '''
    cols/2, rows/2 are important
    '''
    dst = cv2.warpAffine(frame, affine, (cols, rows))

    return dst


def make_transformed_faceset(frame):
    p = "shape.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rect_set = []
    #detect faces in gray scale frame
    rects = detector(gray, 0)
    dst = frame
    #determine facial landmarks over the face
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        #convert facial landmarks to numpy array
        shape = shape_to_np(shape, "int")
        '''
        #loops over (x,y) coordinates for facial landmarks
        for (x,y) in shape:
            cv2.circle(img=frame,center=(x,y),radius=2,color=(0,255,0),thickness=-1)
        '''
        dst = affine_transform(shape, frame) # DO affine_transform
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        rects_prime = detector(gray, 0)

        for (i, rect_prime) in enumerate(rects_prime):
            rect_set.append(rect_prime)

        '''
            written by wooram 2018.08.23
    
            i added affine transform with original source codes about face alignment
    
            detection by dlib is easy and fast
        '''
    return rect_set, dst







