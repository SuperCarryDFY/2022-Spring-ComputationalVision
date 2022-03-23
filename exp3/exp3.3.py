import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread('../chandler.PNG',0)

plt.hist(img.ravel(),256)
plt.title('Histrogram of original image')
plt.show()
# 直方图均衡化
equ = cv.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.title('Histrogram of equal image')
plt.show()

res = np.hstack((img,equ))
cv.imshow("compare",res)
cv.waitKey()
cv.destroyAllWindows()