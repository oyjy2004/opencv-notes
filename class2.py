import cv2

"""读取视频"""
cap = cv2.VideoCapture("video/sunrise.mp4")

"""确定视频是否打开"""
if not cap.isOpened():
    print("ERROR: Cannot open the video")
    exit()

"""显示灰度视频//帧率为60"""
frame_count = 0                 # 记录帧数
while True:
    ret, frame = cap.read()     # 读取视频的一帧
    if not ret:
        print("Error: Cannot read video frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # 转换为灰度图
    cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)  # 创建窗口为可调整大小
    cv2.resizeWindow('Video Frame', 600, 400)  # 设置窗口大小
    cv2.imshow('Video Frame', frame_gray)    # 显示当前帧

    if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
        break

    frame_count += 1

# 释放资源
cap.release()
cv2.destroyAllWindows()
