import cv2
import numpy as np
from functions import cv_show

# 轮廓是连续的，而边界不一定连续
img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
cv_show("IMG_BINARY", img_binary, 0)

# contours, hierarchy = cv2.findContours(img,mode,method)
# mode:轮廓检索模式       method:轮廓逼近方法
# contours:轮廓          hierarchy:层级
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(img, contours, contourIdx, color, thickness)
# contourIdx:轮廓的序号（-1表示所有）    color:线的颜色      thickness:线的宽度
# 必须使用img_draw = img.copy(),否则原图会变!!!不能img_draw = img,因为涉及了地址，可将img和img_draw视为指针
img_draw = img.copy()
cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 1)
cv_show("RES", img_draw, 0)

# 求一个轮廓的面积和周长
contours_120 = contours[120]
S = cv2.contourArea(contours_120)
C = cv2.arcLength(contours_120, True)
print("Area_of_120:", S, "\t\t", "Length_of_120", C)

# 轮廓的近似
# cv.approxPolyDP(cnt, epsilon, closed)
# cnt:要近似的轮廓    epsilon:轮廓近似的阈值     closed:是否闭合
epsilon = 0.01 * C
approx = cv2.approxPolyDP(contours_120, epsilon, True)

img_draw = img.copy()
cv2.drawContours(img_draw, [approx], -1, (0, 0, 255), 1)
cv_show("RES", img_draw, 0)

# 绘制轮廓的边界矩形或者外接圆
# cv2.boundingRect(cnt)
# 找到一个轮廓 (cnt) 的边界矩形，提取该矩形的左上角坐标 (x, y)，以及宽度 w 和高度 h
x1, y1, w, h = cv2.boundingRect(contours_120)
img_draw = img.copy()
cv2.rectangle(img_draw, (x1, y1), (x1+w, y1+h), (0, 0, 255), 1)
cv_show("RES", img_draw, 0)

(x2, y2), r = cv2.minEnclosingCircle(contours_120)
img_draw = img.copy()
cv2.circle(img_draw, (int(x2), int(y2)), int(r), (0, 0, 255), 1)
cv_show("RES", img_draw, 0)
