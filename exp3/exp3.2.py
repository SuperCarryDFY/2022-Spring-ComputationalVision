import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像home_color
home_color = cv.imread("../home_color.png",-1)
home_grey = cv.cvtColor(home_color,cv.COLOR_BGR2GRAY)

# 画出灰度化图像home_gray的灰度直方图，并拼接原灰度图与结果图
# ravel() 将高维数组压成一维
plt.hist(home_grey.ravel(),256)
plt.title('Histrogram of grey image')
plt.savefig('home_grey_hist.png')
plt.show()
home_grey_hist = cv.imread("home_grey_hist.png",0)
# （1）当输入矩阵是uint8类型的时候，此时imshow显示图像的时候，会认为输入矩阵的范围在0-255之间。
# （2）如果imshow的参数是double类型的时候，那么imshow会认为输入矩阵的范围在0-1。
#  所以一定要指明dtype为uint8
home_grey_syn = np.zeros((home_grey.shape[0],home_grey.shape[1]*2),dtype='uint8')
home_grey_syn[:,:home_grey.shape[1]] = home_grey.copy()
home_grey_syn[:,home_grey.shape[1]:] = cv.resize(home_grey_hist,(home_grey.shape[1],home_grey.shape[0]),interpolation=cv.INTER_AREA)

# 画出彩色home_color图像的直方图，并拼接原彩色图与结果图，且与上一问结果放在同一个窗口中显示
color = ('b','g','r')
for id,bgrcolor in enumerate(color):
    plt.hist(home_color[:,:,id].flatten(),bins=256,density=True,color=bgrcolor,alpha=.7)
    plt.title('Histrogram of Color image')
plt.savefig('home_color_hist.png')
plt.show()
home_color_hist = cv.imread("home_color_hist.png",cv.IMREAD_COLOR)
home_color_syn = np.zeros((home_color.shape[0],home_color.shape[1]*2,home_color.shape[2]),dtype='uint8')
home_color_syn[:,:home_color.shape[1],:] = home_color
home_color_syn[:,home_color.shape[1]:,:] = cv.resize(home_color_hist,(home_color.shape[1],home_color.shape[0]),interpolation=cv.INTER_AREA)

ROI_area = np.zeros(home_color.shape,dtype='uint8')
ROI_area[50:100,100:200] = home_color[50:100,100:200]
ROI_mask = np.zeros(home_color.shape,dtype='uint8')
ROI_mask[:][:] = [255,255,255]
ROI_mask[50:100,100:200] = [0,0,0]

for id,bgrcolor in enumerate(color):
    plt.hist(ROI_area[:,:,id].flatten(),bins=256,density=True,color=bgrcolor,alpha=.7)
    plt.title('Histrogram of ROI_area')
plt.savefig('ROI_area.png')
plt.show()
ROI_hist = cv.imread("ROI_area.png",cv.IMREAD_COLOR)
ROI_hist = cv.resize(ROI_hist,(home_color.shape[1],home_color.shape[0]),interpolation=cv.INTER_AREA)
# cv.imshow("roi hist",ROI_hist)

original_mask = np.hstack((home_color,ROI_mask))
roi_and_hist = np.hstack((ROI_area,ROI_hist))
res = np.vstack((original_mask,roi_and_hist))
cv.imshow("res",res)

cv.imshow("home_grey_syn",home_grey_syn)
cv.imshow("home_color_syn",home_color_syn)
cv.waitKey()
cv.destroyAllWindows()