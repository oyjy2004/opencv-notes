import cv2

"""读取图片"""
# imread第二个参数为flags
# cv2.IMREAD_COLOR为彩色读取
# cv2.IMREAD_GRAYSCALE为灰度读取
img = cv2.imread("img/test1.png", cv2.IMREAD_GRAYSCALE)
print(img)
print(img.shape)
print(img.size)
print(img.dtype)
print(type(img))

"""显示图片"""
# imshow第一个参数为窗口名字
# 第二个参数为图像的数组
cv2.imshow("image", img)
# waitKey参数为0  则等待用户按键
# 否则等待指定的时间，单位为ms
cv2.waitKey(0)
cv2.destroyAllWindows()

"""保存图片"""
cv2.imwrite("img/test1_gray.png", img)
