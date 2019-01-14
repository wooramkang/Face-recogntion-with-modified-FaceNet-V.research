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
import pickle
import threading
import time

PADDING = 50
#PADDING = 0

ready_to_detect_identity = True
IS_NEW_DATABASE = False

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

k = 0
face_log = []
face_log_name = []
face_log_count = 0
face_name_gt = {}
face_name_idx_gt = []
face_len = 0
temp_frame = None

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


#FPRmodel = FACE((, 96, 96))
#FRmodel = load_weights_from_FaceNet(FRmodel)
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
FRmodel.load_weights("temp_face.h5")
FPRmodel = FRmodel
#FPRmodel.save_weights("temp_face.h5")
F_P_Rmodel = FPRmodel


def init_tracking(database):

    global face_log_pos, face_log_name, face_log_count, face_name_idx_gt, face_name_gt_idx
    face_log_pos = None
    face_log_name = None
    face_log_count = 0
    face_name_idx_gt = [i for i in database]
    face_name_idx_gt.append("None")
    count = 0
    face_name_gt_idx = {}
    for i in face_name_idx_gt:
        face_name_gt_idx[i] = count
        count = count + 1

def prepare_database():

    database = {}
    img_path = "/home/rd/recognition_reaserch/FACE/Dataset/for_short_train/"

    # load all the images of individuals to recognize into the database
    '''
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        identity = str(identity).split('_')[0]
        #print(identity)
        """ from this, written by wooram 2018. 08. 13
        
        1. is there any easie
        """
        
        if identity in database:
            database[identity].append(img_path_to_encoding(file, F_P_Rmodel))
        else:
            database[identity] = []
    '''
    """
    written by wooram 2018.08.13

    1. think about How to save the embedding metrics

    2. the opposite, How to load

    3. to decide the points of simularity between the embedding metrics and input-pics
    """

    if IS_NEW_DATABASE:
        folders = os.listdir(img_path)
        for name in folders:
            for file in glob.glob(img_path + name + "/*"):
                identity = str(file).split('.')

                if identity[len(identity) - 1] != 'jpg':
                    continue
                '''
                written by wooram kang 2018.09. 14
                 for broken images, you should check the images if it's okay or not
    
                '''
                with open(file, 'rb') as f:
                    check_chars = f.read()[-2:]
                    if check_chars != b'\xff\xd9':
                        print('Not complete image')
                        continue

                identity = name

                if identity in database:
                    database[identity].append(img_path_to_encoding(file, FRmodel))
                else:
                    database[identity] = []
        with open('face_embed.obj', 'wb') as FE:
            pickle.dump(database, FE)
    else:

        with open('face_embed.obj', 'rb') as FE:
            database = pickle.load(FE)

    print("data load done")
    init_tracking(database)

    print(database)
    return database

def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    starting_time = int(time.time())

    while vc.isOpened():
        now = int(time.time())
        diff_time = (now - starting_time) % 2
        _, frame = vc.read()
        #_, _, frame = make_transformed_faceset(frame)
        '''
         written by wooramkang 2018.08. 23
         
         affine transform added
        '''
        img = frame
        # We do not want to detect a new identity while the program is in the process of identifying another person

        if diff_time == 0:
            #face_recognizer_thread = threading.Thread(target=process_frame, args=(img, frame, face_cascade))
            #face_recognizer_thread.start()
            frame = process_frame(img, frame, face_cascade)

        global temp_frame
        if temp_frame is not None:
            frame = temp_frame
            temp_frame = None
        key = cv2.waitKey(100)

        cv2.imshow("preview", frame)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")


def tracking_face(frame, id, face_img = (0,0,10000,10000)):

    global face_log_pos, face_log_name, face_log_count
    global face_name_idx_gt, face_name_gt_idx

    new_id = id

    if (face_log_count > 0) and (face_log_count % 7== 0):
        face_log_pos.pop(0)
        face_log_name.pop(0)
        face_log_count = 6

    face_img = list(face_img)
    height, width, channels = frame.shape

    if face_log_pos is None:
        face_log_pos = []
        face_log_name = []

        face_img[0] = max(0, face_img[0] - PADDING)
        face_img[1] = max(0, face_img[1] - PADDING)
        face_img[2] = min(width, face_img[2] + PADDING)
        face_img[3] = min(height, face_img[3] + PADDING)

        face_log_pos.append(face_img)
        face_log_name.append([id])
        face_log_count = face_log_count + 1

        #print("++++++++")
        #print(face_log_count)

    else:
        t = face_img
        for idx in range(len(face_log_pos)):
            f = face_log_pos[idx]

            if t[0] >= f[0]:
                if t[1] >= f[1]:
                    if t[2] <= f[2]:
                        if t[3] <= f[3]:

                            face_img[0] = max(0, face_img[0] - PADDING)
                            face_img[1] = max(0, face_img[1] - PADDING)
                            face_img[2] = min(width, face_img[2] + PADDING)
                            face_img[3] = min(height, face_img[3] + PADDING)

                            face_log_pos[idx] = face_img
                            face_log_name[idx].append(id)
                            face_log_count = face_log_count + 1

                            zoom_area = [face_name_gt_idx[i] for i in face_log_name[idx]]
                            count_list = [0 for i in range(len(face_name_idx_gt))]

                            for idx_area in zoom_area:
                                count_list[idx_area] = count_list[idx_area] + 1

                            max_id = count_list.index(max(count_list))

                            #print("++___++")
                            #print(face_log_count)

                            return face_name_idx_gt[max_id]

        face_log_pos = None
        face_log_name = None
        face_log_count = 0

    return new_id

def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global k
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    for (x, y, w, h) in faces:

        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        # part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        part_image = img[y1:y2, x1:x2]
        part_image = cv2.resize(part_image, (96, 96))
        cv2.imwrite('next/' + str(k) + '.jpg', part_image)
        k = ((k+1) % 1000)
        img = cv2.rectangle(frame, (x1, y1), (x2, y2),(255,0,0),2)

        identity = find_identity(frame, x1, y1, x2, y2)
        identity = tracking_face(frame, identity, (x1, y1, x2, y2))
        print("final_id")
        print(str(identity))
        print("=======")

        if identity != "None":
            cv2.putText(img, identity, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)

    global temp_frame
    temp_frame = img
    return img


def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1______________
    |                  |
    |                  |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head

    #part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    part_image = frame[y1:y2, x1:x2]

    return who_is_it(part_image, database, FRmodel)


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

    encoding = img_to_encoding(image, model)
    encoding = encoding[0]

    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.

    max = 0
    max_idx = 0

    Threshold =0.5
    name_dist = {}

    for name in database:
        name_mindist = 100
        for query in database[name]:
            dist = np.linalg.norm(query - encoding)

            #print('distance for %s is %s' % (name, dist))
            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < name_mindist:
                name_mindist = dist

        if name_mindist < min_dist:
            min_dist = name_mindist
            identity = name
        name_dist[name] = name_mindist
    list_dist = []
    for n in name_dist:
        list_dist.append(name_dist[n])
    list_dist.sort()

    list_name = []
    for n in name_dist:
        for i in list_dist:
            if i == name_dist[n]:
                list_name.append(n)
                break

    print(name_dist)
    print(list_dist)

    '''                
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name

    '''
    if min_dist >= Threshold:
        identity = "None"
    print("=======")
    print("guess_trial")
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
