# SIFT matching using OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im_gray1 = cv2.imread('../inputs/sift_input1.jpg', 0)
im_gray2 = cv2.imread('../inputs/sift_input2.jpg', 0)

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(im_gray1, None)  # des是描述子
kp2, des2 = sift.detectAndCompute(im_gray2, None)  # des是描述子

# hmerge = np.hstack((im_gray1, im_gray2)) #水平拼接
# cv2.imshow("gray", hmerge) #拼接显示为grayim_
# cv2.waitKey(0)

img_sift_kp_1 = cv2.drawKeypoints(im_gray1, kp1, im_gray1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
img_sift_kp_2 = cv2.drawKeypoints(im_gray2, kp2, im_gray2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

# hmerge = np.hstack((img3, img4)) #水平拼接
# cv2.imshow("point", hmerge) #拼接显示为gray
# cv2.waitKey(0)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# 调整ratio，

matches = sorted(matches, key=lambda x: x[0].distance)

img_most50 = cv2.drawMatchesKnn(im_gray1, kp1, im_gray2, kp2, matches[:50], None, flags=2)
img_least50 = cv2.drawMatchesKnn(im_gray1, kp1, im_gray2, kp2, matches[-50:], None, flags=2)
# cv2.imshow("BFmatch", img_least50)
# cv2.waitKey(0)

##########################################################################################

# Keypoint maps
cv2.imwrite('../results/ex2d_sift_input1.jpg', np.uint8(img_sift_kp_1))
cv2.imwrite('../results/ex2d_sift_input2.jpg', np.uint8(img_sift_kp_2))

# Feature Matching outputs
cv2.imwrite('../results/ex2d_matches_least50.jpg', np.uint8(img_least50))
cv2.imwrite('../results/ex2d_matches_most50.jpg', np.uint8(img_most50))
