import cv2
import numpy as np
src = cv2.imread('/home/lehaisonmath6/Desktop/anhbienso.png')
# src = src[10:,40:src.shape[1]-40]
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
retval, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(thresh,kernel,iterations=1)
# kerne2 = np.ones((5,5),np.uint8)
# erosion = cv2.erode(dilate,kernel,iterations = 1)

cv2.imshow("anh nhi phan",dilate)

_,contours, _  = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for con in contours:
    x,y,w,h = cv2.boundingRect(con)
    if w > 20 and h > 20 and float(h)/w > 1.8 and w*h > 30000:
        cv2.rectangle(src,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('and dest',src)
cv2.waitKey(0)