import cv2
import numpy as np

"""
    written by wooram 2018.08. 17
"""


def to_Lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def median_filter(img):
    return cv2.medianBlur(img, 51)

def to_negative(img):
    return ~img

def preprocessing(img):
    height, width = img.shape[:2]

    if height % 2 == 1:
        height = height +1

    if width % 2 == 1:
        width = width +1

    """
    if height % 20 != 0:
        height = height + (height %20)

    if width % 20 != 0:
        width = width + (width % 20)
    """

    img= cv2.resize(img, (width, height))
    img = to_Lab(img)
    _, a, b = cv2.split(img)
    img = median_filter(img)
    img = to_negative(img)
    lumin, _, _ = cv2.split(img)
    img = cv2.merge((lumin, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img


"""
written by wooram 18.08.14 

"""