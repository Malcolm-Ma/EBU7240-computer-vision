# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/Img01.jpg')
im2 = cv2.imread('../inputs/Img02.jpg')

im_gray1 = cv2.imread('../inputs/Img01.jpg', 0)
im_gray2 = cv2.imread('../inputs/Img02.jpg', 0)

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#


surf = cv2.xfeatures2d.SIFT_create()  # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
kp1, des1 = surf.detectAndCompute(im_gray1, None)  # 查找关键点和描述符
kp2, des2 = surf.detectAndCompute(im_gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
# 提取优秀的特征点
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        good.append(m)
src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
H = cv2.findHomography(src_pts, dst_pts)  # 生成变换矩阵
H1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
h, w = im_gray1.shape[:2]
h1, w1 = im_gray2.shape[:2]
shft = np.array([[1.0, 0, w], [0, 1.0, 0], [0, 0, 1.0]])
M = np.dot(shft, H[0])  # 获取左边图像到右边图像的投影映射关系
M1 = np.dot(shft, H1[0])  # 获取左边图像到右边图像的投影映射关系

panorama_noRANSAC = cv2.warpPerspective(im1, M, (w + w1, h))  # 透视变换，新图像可容纳完整的两幅图
panorama_RANSAC = cv2.warpPerspective(im1, M1, (w + w1, h))  # 透视变换，新图像可容纳完整的两幅图
# cv2.imshow('tiledImg1', panorama_RANSAC)  # 显示，第一幅图已在标准位置
panorama_noRANSAC[0:h, w:w * 2] = im2  # 将第二幅图放在右侧
panorama_RANSAC[0:h, w:w * 2] = im2  # 将第二幅图放在右侧
# cv2.imwrite('tiled.jpg',dst_corners)

# cv2.imshow("hh",panorama_RANSAC)
# cv2.waitKey(0)


##########################################################################################

cv2.imwrite('../results/ex3a_stitched_noRANSAC.jpg', panorama_noRANSAC)
cv2.imwrite('../results/ex3a_stitched_RANSAC.jpg', panorama_RANSAC)
