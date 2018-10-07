import faceutil
import numpy as np
import cv2
import socket
import sys
from threading import Thread
import traceback
from multiprocessing import Pool
import time

list_known_encoding,names = faceutil.load_data_encoding_label('./vecdata/*')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def multi_process_face(params):
    global list_known_encoding,names
    roi_color = params[0]
    x = params[1]
    y = params[2]
    rs  = faceutil.face_recognition_ut_v2(roi_color,list_known_encoding,names,x,y)
    return rs 

cap = cv2.VideoCapture('testface.webm')
sum_fps = 0
num_fps = 0
while(True):
    start = time.time()
    ret , frame = cap.read()
    if( ret == False):
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    list_parameter = []
    for (x,y,w,h) in faces:
        roi_color = np.copy(frame[y:y+h, x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # name = faceutil.face_recognition_ut(roi_color,list_known_encoding,names)
        # cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0))
        params = (roi_color,x,y)
        list_parameter.append(params)
    ls_rs = []
    p = Pool(4)
    ls_rs = p.map(multi_process_face,list_parameter)
    p.close()
    for x,y,name in ls_rs:
        cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0))
    end = time.time()
    fps = 1 / (end - start)
    sum_fps += fps
    num_fps +=1
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1)&0xFF== ord('q')):
        # p.close()
        break
avg = float(sum_fps)/num_fps
print('fps avg: '+ str(avg))
cap.release()


