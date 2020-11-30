# K-Means Clustering
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im = cv2.imread('../inputs/baboon.jpg')
# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
tmp = np.zeros([im.shape[0], im.shape[1]], np.float32)
tmp1 = np.zeros([im.shape[0], im.shape[1]], np.float32)
for i in range(0, im.shape[0]):
    tmp[i][:] = i / 1.2
for j in range(0, im.shape[1]):
    tmp1[:][j] = j / 1.2
im = np.float32(im)


def k_mean(k, xy):
    if xy:
        (b, g, r) = cv2.split(im)
        feature = cv2.merge([b, g, r, tmp, tmp1])
        Z = feature.reshape((-1, 5))
    else:
        feature = im
        Z = feature.reshape((-1, 3))

    # 将数据转化为np.float32
    Z = np.float32(Z)
    # 定义终止标准 聚类数并应用k均值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 现在将数据转化为uint8, 并绘制原图像
    center = np.uint8(center)
    res = center[label.flatten()]

    if xy:
        res = res[:, :3]
    res2 = res.reshape((im.shape))
    cv2.imshow('res2', res2)
    cv2.waitKey(100)
    return res2


result_rgb_2 = k_mean(2, False)
result_rgbxy_2 = k_mean(2, True)
result_rgb_4 = k_mean(4, False)
result_rgbxy_4 = k_mean(4, True)
result_rgb_8 = k_mean(8, False)
result_rgbxy_8 = k_mean(8, True)
##########################################################################################
cv2.imwrite('../results/ex3d_rgb_2.jpg', result_rgb_2)
cv2.imwrite('../results/ex3d_rgbxy_2.jpg', result_rgbxy_2)
cv2.imwrite('../results/ex3d_rgb_4.jpg', result_rgb_4)
cv2.imwrite('../results/ex3d_rgbxy_4.jpg', result_rgbxy_4)
cv2.imwrite('../results/ex3d_rgb_8.jpg', result_rgb_8)
cv2.imwrite('../results/ex3d_rgbxy_8.jpg', result_rgbxy_8)
