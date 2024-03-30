import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, (512, 512))
cv_show("IMG", img, 0)
print(img.shape)

"""Gauss pyramid"""
down = cv2.pyrDown(img)
cv_show("DOWN", down, 0)
print(down.shape)

down_up = cv2.pyrUp(down)
cv_show("DOWN_UP", down_up, 0)
print(down_up.shape)

res = np.hstack((img, down_up))
cv_show("ALL", res, 0)

"""Laplacian pyramid"""
no1 = img - down_up
no2 = down - cv2.pyrUp(cv2.pyrDown(down))
cv_show("NO1", no1, 0)
cv_show("NO2", no2, 0)
print(no1.shape)
print(no2.shape)
