import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test1.png", cv2.IMREAD_GRAYSCALE)

"""Canny流程"""
# 高斯滤波
# sobel算子算梯度
# 非极大值像素梯度抑制，与梯度方向的相邻两个像素点比较，最大则保留（要用线性插值法）
# 阈值滞后处理 minval以下舍去，maxval保留，之间的进行下一步
# 孤立弱边缘抑制 若与强边界点相连，则保留该若边界点，否则舍去
v1 = cv2.Canny(img, 50, 100)
v2 = cv2.Canny(img, 150, 200)

res = np.hstack((v1, v2))
cv_show("ALL", res, 0)
