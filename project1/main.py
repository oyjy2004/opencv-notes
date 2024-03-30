import cv2
import numpy as np


def cv_show(name, img, time):
    """show the image"""
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def sort_contours(cnts, method='left-to-right'):
    """sort the contours we get as the given order"""
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if sort in reverse
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    # handle if sort against y rather than x of the bounding box
    if method == 'bottom-to-top' or method == 'top-to-bottom':
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


"""handle the module"""
# 二值化模板, 将背景变为黑色!!!
module = cv2.imread("F:\python_work\opencv_learn\project1\module.png", cv2.IMREAD_COLOR)
module_gray = cv2.cvtColor(module, cv2.COLOR_BGR2GRAY)
ret, module_bio = cv2.threshold(module_gray, 127, 255, cv2.THRESH_BINARY_INV)

# 读取模板轮廓
# 只读取外轮廓, 且只保留终点坐标
module_contours, hierarchy = cv2.findContours(module_bio, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_draw = module.copy()
cv2.drawContours(img_draw, module_contours, -1, (0, 0, 255), 1)
cv_show("ALL", img_draw, 0)

# 将轮廓从左至右排序
(module_contours, boundingboxes) = sort_contours(module_contours, method="left-to-right")
img_draw = module.copy()
cv2.drawContours(img_draw, module_contours, 5, (0, 0, 255), 1)
cv_show("ALL", img_draw, 0)

# 将轮廓信息存入字典中
digit = {}
for i, c in enumerate(module_contours):
    x, y, w, h = boundingboxes[i]
    theslice = module_bio[y:y+h, x:x+w]
    theslice = cv2.resize(theslice, (64, 90))
    digit[i] = theslice

cv_show("TEST", digit[5], 0)

"""handle the card""" 
# 统一图片尺寸
thecard = cv2.imread("F:\python_work\opencv_learn\project1\card3.jpg", cv2.IMREAD_COLOR)
thecard = cv2.resize(thecard, (742, 461))
thecard_gray = cv2.cvtColor(thecard, cv2.COLOR_BGR2GRAY)
cv_show("CARD", thecard_gray, 0)

# 用礼帽/高帽提取图片较亮的部分
kerner = np.ones((5, 5), dtype = np.uint8)
thecard_tophat = cv2.morphologyEx(thecard_gray, cv2.MORPH_TOPHAT, kerner)
cv_show("CARD_TOPHAT", thecard_tophat, 0)

# 用sobel算子求梯度
thecard_sobel = cv2.Sobel(thecard_tophat, cv2.CV_64F, 1, 0, ksize=3)
thecard_sobel = cv2.convertScaleAbs(thecard_sobel)
cv_show("CARD_SOBEL", thecard_sobel, 0)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
kerner = np.ones((10, 20), dtype = np.uint8)
thecard_close = cv2.morphologyEx(thecard_sobel, cv2.MORPH_CLOSE, kerner)
cv_show("CARD_CLOSE", thecard_close, 0)

# 通过OTSU二值化图像
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
ret, thecard_otsu = cv2.threshold(thecard_close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv_show("CARD_OTSU", thecard_otsu, 0)

# 再通过闭操作优化图像
kerner = np.ones((5, 5), dtype = np.uint8)
thecard_close2 = cv2.morphologyEx(thecard_otsu, cv2.MORPH_CROSS, kerner)
cv_show("CARD_CLOSE2", thecard_close2, 0)

# 得到轮廓
card_contours, hierarchy = cv2.findContours(thecard_close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_draw = thecard.copy()
cv2.drawContours(img_draw, card_contours, -1, (0, 0, 255), 1)
cv_show("CARD_DRAW", img_draw, 0)

# 开始筛选轮廓， 符合条件的留下来
numberlist = []
for i, c in enumerate(card_contours):
    (x, y, w, h) = cv2.boundingRect(c)
    wdh = w / h
    # print(wdh, w)
    if wdh > 2.9 and wdh < 3.5:
        if w > 90 and w < 150:
            numberlist.append((x, y, w, h))
    
# 查看筛选效果并对轮廓进行排序
numberlist = sorted(numberlist, key=lambda b: b[0], reverse=False)
img_draw = thecard.copy()
for x, y, w, h in numberlist:    
    cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 0, 255), 1)
cv_show("CARD_DRAW", img_draw, 0)

# 用来储存输出结果
output = ""

# 遍历四个数字单元，将一个数字单元的每个数字进行单独分析
for (i, (x, y, w, h)) in enumerate(numberlist):
    # 获得切片
    theslice = thecard[y-5:y+h+5, x-5:x+w+5]
    theslice_gray = thecard_gray[y-5:y+h+5, x-5:x+w+5]
    ret, theslice_otsu = cv2.threshold(theslice_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 获得切片的轮廓
    slice_contours, hierarchy = cv2.findContours(theslice_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (slice_contours, boundingboxes) = sort_contours(slice_contours, method="left-to-right")
    img_draw = theslice.copy()
    cv2.drawContours(img_draw, slice_contours, -1, (0, 0, 255), 1)
    cv_show("CARD_DRAW", img_draw, 0)
    
    # 遍历切片中的每个数字
    for cnt in slice_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        the_number = theslice_otsu[y:y+h, x:x+w]
        the_number = cv2.resize(the_number, (64, 90))
        cv_show("CARD_DRAW", the_number, 0)
        # 对每个数字进行模板匹配
        scores = []
        for (i, number_module) in digit.items():
            res = cv2.matchTemplate(the_number, number_module, cv2.TM_CCOEFF_NORMED)
            minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
            scores.append(maxval)
            
        output = output + (str(np.argmax(scores)))
    
    output = output + "\t"
    
# print the answer
print(output)
    