from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    height, width = shape
    steer_matrix_left = np.zeros(shape)
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                steer_matrix_left[y, x] = -0.3 * (x / (width // 2))
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    height, width = shape
    steer_matrix_right = np.zeros(shape)
    for y in range(height):
        for x in range(width):
            if x > width // 2:
                steer_matrix_right[y, x] = 0.1 - (0.1 * ((x - (width // 2)) / (width // 2)))
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # TODO: implement your own solution here
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask_ground = np.ones(img.shape, dtype=np.uint8) 
    mask_ground[:h//2,:] = 0

    sigma = 10

    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 0 
    
    mask_mag = (Gmag > threshold)

    white_lower_hsv = np.array([0, 0, 100])         
    white_upper_hsv = np.array([179, 60, 255])   
    yellow_lower_hsv = np.array([15, 60, 60])        
    yellow_upper_hsv = np.array([35, 255, 255])  

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white


    return mask_left_edge, mask_right_edge
