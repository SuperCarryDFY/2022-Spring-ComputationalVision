import cv2 as cv
from PIL import Image,ImageDraw,ImageFont
import numpy as np

# 加载彩色灰度图像
img = cv.imread('chandler.PNG',-1)
cv2img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
pilimg = Image.fromarray(cv2img)

draw = ImageDraw.Draw(pilimg)
font = ImageFont.truetype('simhei.ttf',20,encoding='utf-8')
draw.text((450,80),'19120199-戴枫源',(255,255,0),font=font)

cv2charimg = cv.cvtColor(np.array(pilimg),cv.COLOR_RGB2BGR)
cv.imshow('image',cv2charimg)

k = cv.waitKey(0)
if k== 27:
    cv.destroyAllWindows()
elif k == ord('s'):
    cv.imwrite('dandler.PNG',cv2charimg)
    cv.destroyAllWindows()

