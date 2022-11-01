"""
https://code.adonline.id.au/cmyk-in-python/
"""

import numpy as np
import cv2


def get_channels_CMYK(img):
    """
    Takes an RGB image, converts it to CMYK and splits the channels saving each as a greyscale image.
    :param img: matplotlib.imread result
    :return: The split grayscale images.
    """

    # Create float
    bgr = img.astype(float) / 255.

    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = (1 - bgr[..., 2] - K) / (1 - K)
        M = (1 - bgr[..., 1] - K) / (1 - K)
        Y = (1 - bgr[..., 0] - K) / (1 - K)

    # TODO testing code
    K = 1 - K
    C = 1 - C
    M = 1 - M
    Y = 1 - Y

    # TODO until here testing

    # Convert the input BGR image to CMYK colorspace
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)

    # Split CMYK channels
    Y, M, C, K = cv2.split(CMYK)

    np.isfinite(C).all()
    np.isfinite(M).all()
    np.isfinite(K).all()
    np.isfinite(Y).all()

    return Y, M, C, K

    # # Save channels
    # cv2.imwrite('C:/path/to/C.jpg', C)
    # cv2.imwrite('C:/path/to/M.jpg', M)
    # cv2.imwrite('C:/path/to/Y.jpg', Y)
    # cv2.imwrite('C:/path/to/K.jpg', K)


def get_channels_CMY(img):
    """
    Takes an RGB image, converts it to CMY and splits the channels saving each as a greyscale image.
    :param img: matplotlib.imread result
    :return: The split grayscale images.
    """

    # Create float
    bgr = img.astype(float) / 255.

    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = 1 - bgr[..., 2]
        M = 1 - bgr[..., 1]
        Y = 1 - bgr[..., 0]

    # # TODO testing code
    # K = 1 - K
    # C = 1 - C
    # M = 1 - M
    # Y = 1 - Y
    #
    # # TODO until here testing

    # Convert the input BGR image to CMYK colorspace
    CMY = (np.dstack((C, M, Y)) * 255).astype(np.uint8)

    # Split CMYK channels
    Y, M, C = cv2.split(CMY)

    np.isfinite(C).all()
    np.isfinite(M).all()
    np.isfinite(Y).all()

    return Y, M, C

    # # Save channels
    # cv2.imwrite('C:/path/to/C.jpg', C)
    # cv2.imwrite('C:/path/to/M.jpg', M)
    # cv2.imwrite('C:/path/to/Y.jpg', Y)
    # cv2.imwrite('C:/path/to/K.jpg', K)