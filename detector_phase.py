import cv2 as cv
import numpy as np
import os

from parameters import *
import utils


def detect_phase(image_path):
    img = np.copy(image_path)
    box = utils.find_circles(img, mg_ratio=0.4, n_circles=1)
    for b in box:
        crop = utils.crop_image(image_path, b)
        resized_im = utils.resize_image(crop)
    # cv.imwrite("Detector_samples/detection_phase_steps/final1.png", resized_im*255)
    return resized_im


for filename in os.listdir(DETECTOR_SAMPLES_DIR + "/original"):
    image = cv.imread(DETECTOR_SAMPLES_DIR + "/original/" + filename)
    res_im = detect_phase(image)
    cv.imshow("Detector phase", res_im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(DETECTOR_SAMPLES_DIR + "/detected/" + filename, 255 * res_im)
# break
