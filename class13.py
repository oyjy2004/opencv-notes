import cv2
import numpy as np
from functions import cv_show
from matplotlib import pyplot as plt

img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
# print(img.shape)

# creat the mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[225:425, 225:435] = 1

# cv2.bitwise_and(iamge,image,mask=mask)
# input1:需要相与的图像1
# input2:需要相与的图像2
# input3:需要相与的区域
img_masked = cv2.bitwise_and(img, img, mask=mask)

cv_show("RES", img_masked, 0)
