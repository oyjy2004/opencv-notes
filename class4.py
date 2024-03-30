import cv2
import numpy as np
from functions import cv_show

"""滤波——图像平滑处理"""
img = cv2.imread("img/test_noise.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

# 均值滤波
# (3, 3)表示9个元素取均值
blur = cv2.blur(img, (3, 3))

# 方框滤波
# 基本与均值滤波相似，可选择是否归一化
# 若不归一化，容易超过255而发生截断
box1 = cv2.boxFilter(img, -1, (3, 3), normalize=True)
box2 = cv2.boxFilter(img, -1, (3, 3), normalize=False)

# 高斯滤波
# 更加重视该点附近的像素点（权重更大）
# 第三个参数为高斯函数的标准差
guess = cv2.GaussianBlur(img, (3, 3), 1)

# 中值滤波
# 采取该点周围的中值像素点
# 第三个参数为正方形边长
median = cv2.medianBlur(img, 3)

# 图像显示
res1 = np.hstack((img, blur, box1))
res2 = np.hstack((box2, guess, median))
res = np.vstack((res1, res2))

cv_show("ALL", res, 0)
