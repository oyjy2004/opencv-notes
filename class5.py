import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test_rag.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)

"""腐蚀去毛刺"""
# 生成卷积核
kernel = np.ones((3, 3), dtype=np.uint8)
# 进行腐蚀操作
# 在卷积核大小中对图片像素点进行卷积
# 取该图像中（3 * 3）区域内的最小值
erosion1 = cv2.erode(img, kernel, iterations=1)
erosion2 = cv2.erode(img, kernel, iterations=2)

res = np.hstack((img, erosion1, erosion2))
cv_show("ALL", res, 0)

"""膨胀回复粗细"""
# 在卷积核大小中对图片像素点进行卷积
# 取该图像中（3 * 3）区域内的最大值
dilate1 = cv2.dilate(erosion2, kernel, iterations=1)
dilate2 = cv2.dilate(erosion2, kernel, iterations=2)

res = np.hstack((erosion2, dilate1, dilate2))
cv_show("ALL", res, 0)
