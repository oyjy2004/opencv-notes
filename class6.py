import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test_rag.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)

"""开运算，先腐蚀再膨胀"""
# 大部分毛刺已经消除，且字体信息也没有发生变化
kerner = np.ones((3, 3), dtype=np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kerner)

"""闭运算，先膨胀再腐蚀"""
# 字体不改变的前提下，我们把字体缺陷信息补全。
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kerner)

"""梯度计算，膨胀图-腐蚀图"""
# 形成了一个空心的字体样式
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kerner)

"""高帽，原始图像 - 开运算结果"""
# 显示毛刺信息
top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kerner)

"""黑帽，信息闭运算结果 - 原始图像"""
# 显示缺陷信息
black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kerner)

res1 = np.hstack((img, opening, closing))
res2 = np.hstack((gradient, top_hat, black_hat))
res = np.vstack((res1, res2))
cv_show("ALL", res, 0)
