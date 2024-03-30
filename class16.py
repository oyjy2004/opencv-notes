# 傅里叶变换是将图像从空间域转换到频率域(将图像的灰度分布函数变换为图像的频率分布函数)
# 傅里叶逆变换是将图像的频率分布函数变换为灰度分布函数(将图像从频率域转换到空间域)
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("img/lena.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.6, fy=0.6)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""傅里叶变化"""
# 进行傅里叶变化时，要将数据转成float32类型
img_float32 = np.float32(img_gray)

# 进行傅里叶变化
# cv2.DFT_COMPLEX_OUTPUT 执行一维或二维复数阵列的逆变换，结果通常是相同大小的复数数组
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

# 将频谱低频从左上角移动到中心位置
dft_shift = np.fft.fftshift(dft)

# 频谱图像双通道复数转换为0~255区间
# cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])求复数的模
# dft_shift[:, :, 0]为实部, dft_shift[:, :, 1]为虚部
result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

"""!!!该操作具有隐患, uint8会越界！！！"""
# 将result转换成uint8便于显示
# result = np.uint8(result)
#
# res1 = np.hstack((img_gray, result))
# ShowImage("RES", res1, 0)

"""高通滤波"""
# 高通滤波器是指通过高频的滤波器，衰减低频而通过高频，常用于增强尖锐的细节，但会导致图像的对比度会降低
# 生成掩码
rows, cols = img_gray.shape
mask = np.ones(dft.shape, np.uint8)
mask[int(rows/2) - 30:int(rows/2) + 30, int(cols/2) - 30:int(cols/2) + 30, :] = 0

idft = dft_shift * mask
idft_shift = np.fft.ifftshift(idft)
img_low = cv2.idft(idft_shift)
img_low = cv2.magnitude(img_low[:, :, 0], img_low[:, :, 1])
# print(img_low)
# img_low = np.uint8(img_low)

plt.subplot(131), plt.imshow(img_gray, cmap='gray')
plt.title('Input image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(result, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_low, cmap='gray')
plt.title('inverse Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()

"""低通滤波"""
# 低通就是保留图像中的低频成分，过滤高频成分，低通表示图像变换缓慢的灰度分量，使图像模糊
# 生成掩码
rows, cols = img_gray.shape
mask = np.zeros(dft.shape, np.uint8)
mask[int(rows/2) - 30:int(rows/2) + 30, int(cols/2) - 30:int(cols/2) + 30, :] = 1

idft = dft_shift * mask
idft_shift = np.fft.ifftshift(idft)
img_low = cv2.idft(idft_shift)
img_low = cv2.magnitude(img_low[:, :, 0], img_low[:, :, 1])
# print(img_low)
# img_low = np.uint8(img_low)

plt.subplot(131), plt.imshow(img_gray, cmap='gray')
plt.title('Input image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(result, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_low, cmap='gray')
plt.title('inverse Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()
