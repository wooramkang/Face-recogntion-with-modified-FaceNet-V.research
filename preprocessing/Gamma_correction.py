

#Gamma correction
from fr_utils import *
from preprocessing.LAB_luminance import *


def preprocessing_gamma_negative(img):
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

    Gamma = 5
    invGamma = 1.0 / Gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    lumin_prime=cv2.LUT(lumin, table)
    img = cv2.merge((lumin_prime, a, b))
    img_prime=to_negative(img)
    lumin_prime, _, _ = cv2.split(img_prime)
    img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img

def preprocessing_gamma(img):
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

    Gamma = 5
    invGamma = 1.0 / Gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    lumin_prime=cv2.LUT(lumin, table)
    img = cv2.merge((lumin_prime, a, b))
    img_prime=to_negative(img)
    lumin_prime, _, _ = cv2.split(img_prime)
    img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img