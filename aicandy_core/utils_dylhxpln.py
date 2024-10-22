import cv2
import numpy as np

def process_image(img, target_width, target_height):
    resized_img = cv2.resize(img, (target_width, target_height))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    return binary_img[None, :, :].astype(np.float32)