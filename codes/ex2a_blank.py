import cv2
import numpy as np


def gaussian_filter_gray(img, kernel, sigma):
    H, W = img.shape
    pad = kernel // 2

    out = np.zeros((H + pad * 2, W + pad * 2))
    out[pad:pad + H, pad:pad + W] = img
    K = np.zeros((kernel, kernel))
    for x in range(-pad, -pad + kernel):
        for y in range(-pad, -pad + kernel):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K = K / (2 * np.pi * sigma * sigma)
    K = K / K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + kernel, x: x + kernel])
    out = out[pad: pad + H, pad: pad + W]
    return out


im_gray = cv2.imread('../inputs/lena.jpg', 0)
im_gray = cv2.resize(im_gray, (256, 256))

result_gf1 = gaussian_filter_gray(im_gray, 5, 1.0)
result_gf2 = gaussian_filter_gray(im_gray, 5, 10.0)
result_gf3 = gaussian_filter_gray(im_gray, 10, 1.0)
result_gf4 = gaussian_filter_gray(im_gray, 10, 10.0)

result_gf1 = np.uint8(result_gf1)
result_gf2 = np.uint8(result_gf2)
result_gf3 = np.uint8(result_gf3)
result_gf4 = np.uint8(result_gf4)

cv2.imwrite('../results/ex2a_gf_5_1.jpg', result_gf1)
cv2.imwrite('../results/ex2a_gf_5_10.jpg', result_gf2)
cv2.imwrite('../results/ex2a_gf_10_1.jpg', result_gf3)
cv2.imwrite('../results/ex2a_gf_10_10.jpg', result_gf4)
