import cv2
from matplotlib import pyplot as plt

img = cv2.imread("img/test1.png", cv2.IMREAD_GRAYSCALE)

# cv2.calcHist(images, channels, mask, histSize, ranges):
# 参数1：要计算的原图，以方括号的传入，如：[img]。
# 参数2：类似前面提到的dims，灰度图写[0]就行，彩色图B/G/R分别传入[0]/[1]/[2]。
# 参数3：要计算的区域ROI，计算整幅图的话，写None。
# 参数4：也叫bins,子区段数目，如果我们统计0-255每个像素值，bins=256；如果划分区间，比如0-15, 16-31…240-255这样16个区间，bins=16。
# 参数5：range,要计算的像素值范围，一般为[0,256)。
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# print(type(hist), hist.shape)
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

img = cv2.imread("img/test1.png", cv2.IMREAD_COLOR)
color = ('b', 'g', 'r')

for num, thecolor in list(enumerate(color)):
    hist = cv2.calcHist([img], [num], None, [256], [0, 256])
    plt.plot(hist, color=thecolor)
    plt.xlim([0, 256])
plt.show()
