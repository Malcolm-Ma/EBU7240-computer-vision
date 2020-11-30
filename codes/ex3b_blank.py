# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/building.jpg')
im2 = cv2.imread('../inputs/YOUR_OWN.jpg')
# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(gray2, 5, 100, apertureSize=3)
lines2 = cv2.HoughLines(edges2, 1, np.pi / 180, 200)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(im1, (x1, y1), (x2, y2), (0, 0, 255), 2)

for line in lines2:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(im2, (x1, y1), (x2, y2), (0, 0, 255), 2)

im_result = im1
im_result2 = im2
##########################################################################################

cv2.imwrite('../results/ex3b_building_hough.jpg', im_result)
cv2.imwrite('../results/ex3b_YOUR_OWN_hough.jpg', im_result2)
