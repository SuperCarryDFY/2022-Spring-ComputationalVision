import numpy as np
import cv2 as cv

img = cv.imread('../chandler.PNG',-1)


rows,cols,way = img.shape

# 平移：x轴平移100像素，y轴平移150像素
M = np.float32([[1,0,100],[0,1,150]])
dst1 = cv.warpAffine(img,M,(cols,rows))

# 缩放：缩放到1024*768；按比例缩小（60%）
dst2_1 = cv.resize(dst1,(1024,768))
dst2 = cv.resize(dst2_1,None,fx=0.6,fy=0.6)

# 翻转：水平翻转，垂直翻转，水平+垂直翻转
dst3 = cv.flip(dst2,-1)

# 旋转：给出旋转中心，旋转角度，对图片旋转
# 将图片相对中心旋转90度
rows4,cols4,way = dst3.shape
M4 = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst4 = cv.warpAffine(dst3,M,(cols4,rows4))

# 缩略：将图片缩小0.5倍，放到原图的左上角
dst5_1 = cv.resize(dst4,None,fx=0.5,fy=0.5)
rows5,cols5 = dst5_1.shape[:2]
dst5 = img[:]
dst5[:rows5,:cols5] = dst5_1


cv.imshow('image',dst5)
cv.waitKey(0)
cv.destroyAllWindows()