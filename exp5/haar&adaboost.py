import cv2 as cv
import numpy as np

[x, y, w, h] = [0, 0, 0, 0]
path1 = '../self_dataset/2.jpg'
path2 = '../ORL_dataset/s2/4.bmp'

frame = cv.imread(path2)
# frame = cv.resize(frame,(0,0),fx=0.25,fy=0.25)
# 载入人脸adaboost分类器，其特征采用haar
face_Cascade = cv.CascadeClassifier("C:/anaconda/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

size = frame.shape[:2]
print(size)
image = np.zeros(size,dtype=np.float32)
image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

# 直方图均衡
image = cv.equalizeHist(image)
im_h,im_w = size
minSize_l = (im_w // 10, im_h //10)
# 设置搜索窗口的比例系数为1.05
# 设置构成检测目标的相邻矩形的最小个数为2
faceRects = face_Cascade.detectMultiScale(image, 1.05, 2)
if len(faceRects):
    for faceRect in faceRects:
        x, y, w, h = faceRect
        cv.rectangle(frame, (x, y), (x+w,y+h), [255, 255, 0], 2)

cv.imshow("detection",frame)
cv.waitKey(0)
