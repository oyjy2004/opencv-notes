import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test1.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)

"""sobel算子"""
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

"""scharr算子"""
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharr = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

"""laplacian算子"""
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# print all
res = np.hstack((sobel, scharr, laplacian))
cv_show("ALL", res, 0)
