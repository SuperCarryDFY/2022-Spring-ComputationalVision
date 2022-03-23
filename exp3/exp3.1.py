import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取一张图片，将其转换为HSV空间
img = cv.imread('../chandler.PNG',-1)
img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow("img",img_hsv)

# 分离原图片RGB通道及转换后的图片HSV通道
B,G,R = cv.split(img)
zeros = np.zeros(img.shape[:2],dtype="uint8")
cv.imshow("B",cv.merge([B,zeros,zeros]))
cv.imshow("G",cv.merge([zeros,G,zeros]))
cv.imshow("R",cv.merge([zeros,zeros,B]))

H,S,V = cv.split(img_hsv)
cv.imshow('H', H)
cv.imshow('S', S)
cv.imshow('V', V)

cv.waitKey()
cv.destroyAllWindows()

# 对RGB三个通道分别画出其三维图
fig = plt.figure(figsize=(12,8))
ax = Axes3D(fig)
zB = img[:,:,0]
zG = img[:,:,1]
zR = img[:,:,2]
x = np.arange(0,828,1)
y = np.arange(0,578,1)
X,Y = np.meshgrid(x,y)

# 这里以画出B通道的三维图为例
# 如果要画出G通道的三维图，只需修改下面的函数中第三个参数为zG即可,R通道同理
ax.plot_surface(X,Y,zB,cmap=plt.get_cmap('rainbow'))
plt.title("B")
plt.show()
