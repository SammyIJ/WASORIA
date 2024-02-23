# /PycharmProjects/Samuel/image_preprocessing.py
import cv2 as cv
import numpy as np

def canny_edge_detection(image_paths, threshold1=100, threshold2=200):
    edge_detected_images = []
    for path in image_paths:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)  # Convert image directly to grayscale
        edges = cv.Canny(img, threshold1, threshold2)
        edge_detected_images.append(edges)
    return edge_detected_images


def erosion(images, kernel_size=(5, 5), iterations=1):
    eroded_images = []
    kernel = np.ones(kernel_size, np.uint8)
    for img in images:
        eroded_img = cv.erode(img, kernel, iterations=iterations)
        eroded_images.append(eroded_img)
    return eroded_images


def dilation(images, kernel_size=(5, 5), iterations=1):
    dilated_images = []
    kernel = np.ones(kernel_size, np.uint8)
    for img in images:
        dilated_img = cv.dilate(img, kernel, iterations=iterations)
        dilated_images.append(dilated_img)
    return dilated_images
