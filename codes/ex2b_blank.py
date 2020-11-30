import cv2
import numpy as np


def bilateral_filter_gray(img, kernel, sigma1, sigma2):
    H, W = img.shape
    pad = kernel // 2

    out = np.zeros((H + pad * 2, W + pad * 2))
    out[pad:pad + H, pad:pad + W] = img
    K = np.zeros((kernel, kernel))
    for x in range(-pad, -pad + kernel):
        for y in range(-pad, -pad + kernel):
            K[y + pad, x + pad] = np.exp(
                -(x ** 2 + y ** 2) / (2 * (sigma1 ** 2)) - (x ** 2 + y ** 2) / (2 * (sigma2 ** 2)))

    # K=K/(2 * np.pi * sigma1 * sigma2)
    K = K / K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + kernel, x: x + kernel])
    out = out[pad: pad + H, pad: pad + W]
    return out


im_gray = cv2.imread('../inputs/cat.png', 0)

result_bf1 = bilateral_filter_gray(im_gray, 10, 30.0, 3.0)
result_bf2 = bilateral_filter_gray(im_gray, 10, 30.0, 30.0)
result_bf3 = bilateral_filter_gray(im_gray, 10, 100.0, 3.0)
result_bf4 = bilateral_filter_gray(im_gray, 10, 100.0, 30.0)
result_bf5 = bilateral_filter_gray(im_gray, 5, 100.0, 30.0)

result_bf1 = np.uint8(result_bf1)
result_bf2 = np.uint8(result_bf2)
result_bf3 = np.uint8(result_bf3)
result_bf4 = np.uint8(result_bf4)
result_bf5 = np.uint8(result_bf5)

cv2.imwrite('../results/ex2b_bf_10_30_3.jpg', result_bf1)
cv2.imwrite('../results/ex2b_bf_10_30_30.jpg', result_bf2)
cv2.imwrite('../results/ex2b_bf_10_100_3.jpg', result_bf3)
cv2.imwrite('../results/ex2b_bf_10_100_30.jpg', result_bf4)
cv2.imwrite('../results/ex2b_bf_5_100_30.jpg', result_bf5)
