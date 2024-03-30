import cv2
import numpy as np
from functions import cv_show
from matplotlib import pyplot as plt

# OpenCV中用cv2.equalizeHist() 实现均衡化:
# 灰度图均衡，直接使用cv2.equalizeHist(gray)
# 彩色图均衡，分别在不同的通道均衡后合并

# 灰度图
img = cv2.imread("img/test2.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.4, fy=0.4)
equ = cv2.equalizeHist(img)
res1 = np.hstack((img, equ))
cv_show("RES1", res1, 0)
# 直方图展示
hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([equ], [0], None, [256], [0, 256])
plt.plot(hist1, color='b')
plt.show()
plt.plot(hist2, color='r')
plt.show()

# 彩色图
img = cv2.imread("img/test2.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.4, fy=0.4)
# 通道分解
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
equ2 = cv2.merge((bH, gH, rH))
# 水平拼接原图和均衡图
res2 = np.hstack((img, equ2))
cv_show("RES2", res2, 0)

# CLAHE 自适应均衡化
# 直方图均衡化是应用于整幅图片的，会导致一些图片部位太亮，导致大部分细节丢失，因此引入自适应均衡来解决这个问题。
# 它在每一个小区域内（默认8×8）进行直方图均衡化。当然，如果有噪点的话，噪点会被放大，需要对小区域内的对比度进行了限制。
# 彩色图同样需要split为r,g,b后均衡，然后merge。
img = cv2.imread("img/test2.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.2, fy=0.2)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
# 将原图、均衡化图、自适应均衡化图并列展示
equ = cv2.equalizeHist(img)
res3 = np.hstack((img, equ, cl1))
cv_show("ALL", res3, 0)

hist3 = cv2.calcHist([cl1], [0], None, [256], [0, 256])
plt.plot(hist3)
plt.show()
