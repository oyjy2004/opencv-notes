import cv2
import numpy as np
from functions import cv_show

img = cv2.imread("img/test_template.png", cv2.IMREAD_COLOR)
# img = cv2.resize(img, None, fx=0.6, fy=0.6)
# img_s = img[30:150, 100:230, :]
img_s = cv2.imread("img/template.png", cv2.IMREAD_COLOR)
print(f'ImgShape:{img.shape}')
print(f'Img_sShape:{img_s.shape}')
cv_show("img", img, 0)
cv_show("SOME", img_s, 0)

# 匹配方法：将匹配图像(a * b)放于原图像(A * B)的左上点开始逐行扫描并且计算误差
# 计算的结果返回在一个矩阵中(A-a+1 * B-b+1)
# cv2.matchTemplate(img, template, method)
# method = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
#           'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
#           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
# 后缀NORMED表示进行归一化
#
# 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED': 计算相关相关系数(大好)
# 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED': 计算相关性(大好)
# 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED':计算平方误差(小好)
res = cv2.matchTemplate(img, img_s, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)     # min_loc and max_loc is (x, y)
print(f'ResShape:{res.shape}')
print(f'MinVal:{min_val}\tMinLoc:{min_loc}')
print(f'MaxVal:{max_val}\tMaxLoc:{max_loc}')

# 画出匹配位置
img_draw = img.copy()
cv2.rectangle(img_draw, max_loc, (max_loc[0]+img_s.shape[1], max_loc[1]+img_s.shape[0]), (0, 0, 255), 2)
cv_show("res", img_draw, 0)

# 匹配多个目标
threshold = 0.95
# np.where:返回一个元组，元素为两个数组，一个装y, 一个装x
loc = np.where(res >= threshold)

img_draw = img.copy()
# 先用loc[::-1]将(y, x)转换为(x, y)
# zip(*loc)将元组解包
for point in list(zip(*loc[::-1])):      # point is (x, y)
    cv2.rectangle(img_draw, point, (point[0] + img_s.shape[1], point[1] + img_s.shape[0]), (0, 0, 255), 2)

cv_show("res", img_draw, 0)
