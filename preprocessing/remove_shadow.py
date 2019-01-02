import numpy as np
import cv2
from PIL import Image
from skimage import exposure


# read an image with shadow...
# and it converts to BGR color space automatically
def to_Lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def median_filter(img):
    return cv2.medianBlur(img, 51)


def to_negative(img):
    return ~img


def preprocessing(img):
    # height, width = img.shape[:2]

    # if height % 2 == 1:
    #    height = height +1

    # if width % 2 == 1:
    #    width = width +1

    """
    if height % 20 != 0:
        height = height + (height %20)

    if width % 20 != 0:
        width = width + (width % 20)
    """

    # img= cv2.resize(img, (width, height))
    img = to_Lab(img)
    _, a, b = cv2.split(img)
    img = median_filter(img)
    img = to_negative(img)
    lumin, _, _ = cv2.split(img)
    img = cv2.merge((lumin, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img


def preprocessing_hist(img):
    # height, width = img.shape[:2]

    # if height % 2 == 1:
    #    height = height +1

    # if width % 2 == 1:
    #    width = width +1
    """
    if height % 20 != 0:
        height = height + (height %24)

    if width % 20 != 0:
        width = width + (width % 24)
    """
    # img= cv2.resize(img, (width, height))
    img = to_Lab(img)
    lumin, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGize=(8, 8))
    lumin_prime = clahe.apply(lumin)
    img = cv2.merge((lumin_prime, a, b))
    # img_prime=to_negative(img)
    # lumin_prime, _, _ = cv2.split(img_prime)
    # img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img


def preprocessing_hist_nagative(img):
    height, width = img.shape[:2]

    if height % 2 == 1:
        height = height + 1

    if width % 2 == 1:
        width = width + 1
    """
    if height % 20 != 0:
        height = height + (height %24)

    if width % 20 != 0:
        width = width + (width % 24)
    """
    img = cv2.resize(img, (width, height))
    img = to_Lab(img)
    lumin, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lumin_prime = clahe.apply(lumin)
    img = cv2.merge((lumin_prime, a, b))
    img_prime = to_negative(img)
    lumin_prime, _, _ = cv2.split(img_prime)
    img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LRGB)

    return img


def convert_to_pil(nimg):
    """ Converts numpy image to PIL image"""
    return Image.fromarray(np.uint8(nimg))


def convert_to_np(img):
    """ Converts PIL image to np image"""
    # img = img.convert(mode='RGB')
    nimg = np.asarray(img)
    nimg.flags.writeable = True
    return nimg


def max_white(img):
    """ Function to apply simple white balance to a PIL image
    Notes:
        From: https://github.com/shunsukeaihara/colorcorrect
    Arguments:
        img (Image): Input image to white balance
    Returns:
        balanced_image (Image): Balanced image
    """

    # Convert PIL image to numpy array
    nimg = convert_to_np(img)

    # Determine value of brightest possible pixel
    if nimg.dtype == np.uint8:
        brightest = float(2 ** 8)
    elif nimg.dtype == np.uint16:
        brightest = float(2 ** 16)
    elif nimg.dtype == np.uint32:
        brightest = float(2 ** 32)
    else:
        brightest = float(2 ** 8)

    # Apply max white
    nimg = np.transpose(nimg, (2, 0, 1))
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest / float(nimg[0].max())), 255)
    nimg[1] = np.minimum(nimg[1] * (brightest / float(nimg[1].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (brightest / float(nimg[2].max())), 255)
    nimg = np.transpose(nimg, (1, 2, 0)).astype(np.uint8)

    # Convert image back to PIL
    balanced_image = convert_to_pil(nimg)
    return balanced_image


def clean_image(img):
    """ Function to adjust contrast if needed and white balance
    Arguments:
        img (Image): Input image to white balance
    Returns:
        balanced_image (Image): Balanced image
    """

    nimg = convert_to_np(img)

    # Determine if image is low contrast and adjust if needed
    low_contrast = exposure.is_low_contrast(nimg)
    if low_contrast:
        # Convert PIL image to numpy array
        nimg = exposure.adjust_gamma(nimg)

    img = convert_to_pil(nimg)

    balanced_image = max_white(img)
    return balanced_image


def preprocessing_gamma(img, erosion):
    img_orig = img
    height, width = img.shape[:2]

    """
    if height % 2 == 1:
        height = height +1

    if width % 2 == 1:
        width = width +1

    if height % 20 != 0:
        height = height + (height %24)

    if width % 20 != 0:
        width = width + (width % 24)
    """
    img = cv2.resize(img, (width, height))
    img = to_Lab(img)
    lumin, a, b = cv2.split(img)

    Gamma = 2.5
    invGamma = 1.0 / Gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    lumin_prime = cv2.LUT(lumin, table)
    img = cv2.merge((lumin_prime, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2LBGR)

    for i in range(height):
        for j in range(width):
            if erosion[i, j, 0] == 255 and erosion[i, j, 1] == 255 and erosion[i, j, 2] == 255:
                img_orig[i, j] = [img[i, j, 0], img[i, j, 1], img[i, j, 2]]

    return img_orig


def remove_shadow_tt(img):
    or_img = img

    # covert the BGR image to an YCbCr image
    y_cb_cr_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2YCrCb)

    # copy the image to create a binary mask later
    binary_mask = np.copy(y_cb_cr_img)

    # get mean value of the pixels in Y plane
    y_mean = np.mean(cv2.split(y_cb_cr_img)[0])

    # get standard deviation of channel in Y plane
    y_std = np.std(cv2.split(y_cb_cr_img)[0])

    # classify pixels as shadow and non-shadow pixels
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):

            if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
                # paint it white (shadow)
                binary_mask[i, j] = [255, 255, 255]
            else:
                # paint it black (non-shadow)
                binary_mask[i, j] = [0, 0, 0]

    # Using morphological operation
    # The misclassified pixels are
    # removed using dilation followed by erosion.
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_mask, kernel, iterations=1)

    # sum of pixel intensities in the lit areas
    spi_la = 0

    # sum of pixel intensities in the shadow
    spi_s = 0

    # number of pixels in the lit areas
    n_la = 0

    # number of pixels in the shadow
    n_s = 0

    # get sum of pixel intensities in the lit areas
    # and sum of pixel intensities in the shadow

    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if erosion[i, j, 0] == 0 and erosion[i, j, 1] == 0 and erosion[i, j, 2] == 0:
                spi_la = spi_la + y_cb_cr_img[i, j, 0]
                n_la += 1
            else:
                spi_s = spi_s + y_cb_cr_img[i, j, 0]
                n_s += 1

    # get the average pixel intensities in the lit areas
    average_ld = spi_la / n_la

    # get the average pixel intensities in the shadow
    average_le = spi_s / n_s

    # difference of the pixel intensities in the shadow and lit areas
    i_diff = average_ld - average_le

    # get the ratio between average shadow pixels and average lit pixels
    # get the ratio between average shadow pixels and average lit pixels
    ratio_as_al = average_ld / average_le

    # added these difference

    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if erosion[i, j, 0] == 255 and erosion[i, j, 1] == 255 and erosion[i, j, 2] == 255:
                y_cb_cr_img[i, j] = [y_cb_cr_img[i, j, 0] + i_diff, y_cb_cr_img[i, j, 1] + ratio_as_al,
                                     y_cb_cr_img[i, j, 2] + ratio_as_al]

    # covert the YCbCr image to the BGR image

    # final_image = cv2.cvtColor(y_cb_cr_img, cv2.COLOR_YCR_CB2BGR)

    final_image = preprocessing_gamma(or_img, erosion)
    # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2YCrCb)
    final_image = np.array(clean_image(final_image))
    final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
    final_image = preprocessing_gamma(final_image, erosion)
    final_image = np.array(clean_image(final_image))
    final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
    # final_image = preprocessing_gamma(final_image, erosion)
    # final_image = np.array(clean_image(final_image))
    final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
    return final_image


def remove_shadow(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []

    for plane in rgb_planes:
        norm_img = plane
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    # result_norm = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)

    return result_norm