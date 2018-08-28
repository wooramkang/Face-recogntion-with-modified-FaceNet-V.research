

#contrast_limited_adaptive_histogram_equalization

import cv2
from preprocessing.LAB_luminance import *
"""
    written by wooram 2018.08. 17
"""


def preprocessing_hist(img):
    #height, width = img.shape[:2]

    #if height % 2 == 1:
    #    height = height +1

    #if width % 2 == 1:
    #    width = width +1
    """
    if height % 20 != 0:
        height = height + (height %24)

    if width % 20 != 0:
        width = width + (width % 24)
    """
    #img= cv2.resize(img, (width, height))
    img = to_Lab(img)
    lumin, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
    '''
    gridsize = [ (4,4) , (8, 8) , None ]
    '''
    lumin_prime = clahe.apply(lumin)
    img = cv2.merge((lumin_prime, a, b))
    #img_prime=to_negative(img)
    #lumin_prime, _, _ = cv2.split(img_prime)
    #img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img

def preprocessing_hist_nagative(img):
    height, width = img.shape[:2]

    if height % 2 == 1:
        height = height +1

    if width % 2 == 1:
        width = width +1
    """
    if height % 20 != 0:
        height = height + (height %24)

    if width % 20 != 0:
        width = width + (width % 24)
    """
    img= cv2.resize(img, (width, height))
    img = to_Lab(img)
    lumin, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lumin_prime = clahe.apply(lumin)
    img = cv2.merge((lumin_prime, a, b))
    img_prime=to_negative(img)
    lumin_prime, _, _ = cv2.split(img_prime)
    img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img