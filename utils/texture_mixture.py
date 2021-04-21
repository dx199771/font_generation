import cv2
import os
import numpy as np
src_img = "./11.png"
texture_img = "./Texture_4.jpg"
im = cv2.imread(src_img)
texture = cv2.imread(texture_img)

s_h, s_w, s_c = im.shape
h, w, c = texture.shape
texture = cv2.resize(texture,(s_h,s_w),cv2.INTER_AREA)


imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


img = cv2.drawContours(texture, contours, -1, 0, 1)

print(len(contours))
for i in range(s_h):
    for j in range(s_w):
        if (cv2.pointPolygonTest(contours[0],(i,j),True))<0 or (cv2.pointPolygonTest(contours[1],(i,j),True))>0:
            img[j,i,0] = 0
            img[j,i,1] = 0
            img[j,i,2] = 0

texture = cv2.resize(img,(512,512),cv2.INTER_AREA)


cv2.imshow("contours",texture)
cv2.waitKey(0)
cv2.destroyWindow()