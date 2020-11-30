import cv2
import numpy as np


def adaptiveThreshold(n, b):
    im = cv2.imread('../inputs/writing_ebu7240.png')
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 185 * 300
    h = img.shape[0]
    w = img.shape[1]
    img_padding = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    res = np.zeros((h, w), dtype=np.int)

    sum = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            sum += img_padding[(i + 2) + i, (j + 2) + j]
    ksize = 2 * n + 1
    num = ksize * ksize
    mean = sum / num

    for i in range(h):
        for j in range(w):
            if (img[i, j] > mean * b):
                res[i, j] = 255
            else:
                res[i, j] = 0
    return res


if __name__ == '__main__':
    output = adaptiveThreshold(2, 0.4)
    cv2.imwrite('../results/ex3c_thres_asdas0.4.jpg', output)
    output = adaptiveThreshold(2, 0.6)
    cv2.imwrite('../results/ex3c_thres_asdas0.6.jpg', output)
    output = adaptiveThreshold(2, 0.8)
    cv2.imwrite('../results/ex3c_thres_asdas0.8.jpg', output)
