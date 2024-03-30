import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test1.png", cv2.IMREAD_GRAYSCALE)

# 用sobel算子近似计算不同方向的梯度
# cv2.Sobel(src,ddepth,dx,dy,ksize)
# dx为x上梯度次数，dy为y上梯度次数
"""计算x梯度"""
# 不考虑负数
sobel_x1 = cv2.Sobel(img, -1, 1, 0, ksize=3)
# 考虑负数
sobel_x2 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x2 = cv2.convertScaleAbs(sobel_x2)
# 注意！！！数据类型不同不能堆叠
res = np.hstack((sobel_x1, sobel_x2))
cv_show("ALL", res, 0)

"""计算y梯度"""
sobel_y1 = cv2.Sobel(img, -1, 0, 1, ksize=3)
sobel_y2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_y2 = cv2.convertScaleAbs(sobel_y2)
res = np.hstack((sobel_y1, sobel_y2))
cv_show("ALL", res, 0)

"""计算整个图形"""
# 直接用sobel函数
sobel1 = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobel1 = cv2.convertScaleAbs(sobel1)
# 用addWeighted函数
sobel2 = cv2.addWeighted(sobel_x2, 0.5, sobel_y2, 0.5, 0)
res = np.hstack((sobel1, sobel2))
cv_show("ALL", res, 0)
