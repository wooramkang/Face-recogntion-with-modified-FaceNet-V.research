from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from utils import *
from inception_blocks_v2 import *
from preprocessing.LAB_luminance import *
import preprocessing.histogram_equalization as hist
import preprocessing.Gamma_correction as gamma
from preprocessing.remove_shadow import *
from Model import *
from detect_landmarks_plus_affineTransform import *
from model_prime import *

#PADDING = 50
PADDING = 0
ready_to_detect_identity = True
#FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FPRmodel = FACE((1, 96, 96))
FPRmodel.load_weights("REALFACE_final_facenn.h5")
F_P_Rmodel = FACE((1, 96, 96))
#F_P_Rmodel = simpler_face_NN_residualnet((3, 96, 96), 64)
F_P_Rmodel = FPRmodel

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss


#FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
#load_weights_from_FaceNet(FRmodel)


def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        identity = str(identity).split('_')[0]
        #print(identity)
        """ from this, written by wooram 2018. 08. 13
        
        1. is there any easier way to read dataset?
        """
        database[identity] = img_path_to_encoding(file, F_P_Rmodel)

    #print(str(database["wooram"]))
    """
    written by wooram 2018.08.13
    
    1. think about How to save the embedding metrics
    
    2. the opposite, How to load
    
    3. to decide the points of simularity between the embedding metrics and input-pics
    """
    return database


def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #F_P_Rmodel.load_weights('FACE_final_facenn.h5')
    #F_P_Rmodel.load_weights('REALFACE_final_facenn.h5')
    while vc.isOpened():
        _, frame = vc.read()
        #_, _, frame = make_transformed_faceset(frame)
        '''
         written by wooramkang 2018.08. 23
         
         affine transform added
        '''
        img = frame
        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            frame = process_frame(img, frame, face_cascade)
            #img = process_frame(img, frame, face_cascade)

        key = cv2.waitKey(100)
        #frame = remove_shadow(frame)
        cv2.imshow("preview", frame)
        #cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")


def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    #faces, frame = make_transformed_faceset(frame)
    '''
        written by wooramkang 2018.08.23    
    
        1. applying affine_transform is damn hard
        
        2. when it's okay, frame-down happends badly
    
    '''
    #frame = #preprocessing(frame)
    """
    by using LAB_luminance 
    preprocessed
    """
    #frame = hist.preprocessing_hist(frame)
    """
        by using CLAHE (Contrast Limited Adaptive Histogram Equalization  
        preprocessed
    """
    #frame = gamma.preprocessing_gamma(frame)
    """
            by using gamma correction  
            preprocessed
    """

    for (x, y, w, h) in faces:
        print("=======")
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            cv2.putText(img, identity, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
            identities.append(identity)

    #key = cv2.waitKey(100)
    #cv2.namedWindow("Face")
    #cv2.imshow("Face", frame)

    if identities != []:
        cv2.imwrite('_'.join(identities) + '.png', img)
    '''     
            #ready_to_detect_identity = False
            #pool = Pool(processes=1)
            # We run this as a separate process so that the camera feedback does not freeze
            #pool.apply_async(welcome_users, [identities])
            
    #written by wooramKang 2018.08.27
    #for multi_processess running, it's not necessary 
    '''

    return img





def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                  |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    #part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    part_image = frame[y1:y2, x1:x2]
    return who_is_it(part_image, database, F_P_Rmodel)


def who_is_it(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    model.load_weights("REALFACE_final_facenn.h5")
    encoding = img_to_encoding(image, model)
    encoding = encoding[0]

    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.

    max = 0
    max_idx = 0
    temp = []
    for l in encoding:
        temp.append(int(l*100))

    encoding = temp
    del temp
    print(encoding)

    t = 0
    for k in encoding:
        if k > max:
            max = k
            max_idx = t

        t = t+1

    '''
    for (name, db_enc) in database.items():
        
        
         
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
       
     '''

    if max > 70:
        identity = max_idx
    else:
        identity = -1

    identity = max_idx
    print(str(identity))
    return str(identity)


'''
    if min_dist > 0.52:
        return None
    else:
        print(str(identity))
        return str(identity)
'''

"""   
    written by wooram 2018.08.14
    
    1. is there better way about desiding the points of max to distinguish
    
    
    written by wooram 2018.08.14
    
    2. if there are lots of people or a group of people, how to tag them and show them

"""


if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)