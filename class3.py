import cv2
import numpy as np
from matplotlib import pyplot as plt
from functions import cv_show

"""图像分通道"""
img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)

# 将图像分为三个通道 b g r
b, g, r = cv2.split(img)

g = g * 0
r = r * 0
# 将三个通道合并
img = cv2.merge((b, g, r))

cv_show("IMAGE", img, 0)


"""图像的缩放"""
img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
# (width, height)：直接指定需要的图像大小 ！！！先width后height
# fx,fy：当设定(width, height)为(0,0)时，fx与fy分别表示图片两个方向上的缩放比例
# img_c = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
img_c = cv2.resize(img, (600, 480))
cv_show("IMG_C", img_c, 0)


"""图像的融合"""
img1 = cv2.imread("img/test2.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
img1_c = cv2.resize(img1, (600, 600))
img2_c = cv2.resize(img2, (600, 600))
res = cv2.addWeighted(img1_c, 0.4, img2_c, 0.6, 0)

res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()


"""图像边界填充"""
img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

top_size, bottom_size, right_size, left_size = 20, 20, 20, 20
# 复制最边上填充
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                               borderType=cv2.BORDER_REPLICATE)
# 以边界为对称轴
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                             borderType=cv2.BORDER_REFLECT)
# 以边界像素点为对称轴
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                                borderType=cv2.BORDER_REFLECT_101)
# 外包装填充
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                          borderType=cv2.BORDER_WRAP)
# 常量填充
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                              borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

# 显示图像
res = np.hstack((replicate, reflect, reflect101, wrap, constant))
cv_show("ALL", res, 0)


"""图像的阈值处理"""
img = cv2.imread("img/test1_gray.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

# 小于置0 大于置255
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 小于置255 大于置0
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# 大于置为阈值 小于不变
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# 大于不变 小于置0
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# 大于置0 小于不变
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

# 显示图像
res = np.hstack((thresh1, thresh2, thresh3, thresh4,thresh5))
cv_show("ALL", res, 0)
