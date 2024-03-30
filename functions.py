import cv2


def cv_show(str, image, time):
    """图片显示函数"""
    cv2.imshow(str, image)
    cv2.waitKey(time)
    cv2.destroyAllWindows()
