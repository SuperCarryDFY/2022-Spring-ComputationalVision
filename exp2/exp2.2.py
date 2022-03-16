import numpy as np
import cv2 as cv
import math

# 读入为灰度图像
img = cv.imread('../chandler.PNG',0)

rows,cols = img.shape

# 调整灰度图片为正方形（边长不小于500像素）
img2 = cv.resize(img,(min(rows,cols),min(rows,cols)))

# 用圆形掩膜对图片进行切片，并保存切片后的图像
def distance(a,b,x,y):
    return math.sqrt(math.pow((a-x),2)+math.pow((b-y),2))

d = min(rows,cols)
o_i,o_j = d/2,d/2
for i in range(d):
    for j in range(d):
        if distance(i,j,o_i,o_j)>d/2:
            img2[i,j] = 255

cv.imshow('image',img2)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
elif k == ord('s'):
    cv.imwrite('exp2.2.PNG',img2)
    cv.destroyAllWindows()
