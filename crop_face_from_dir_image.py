import cv2
import glob
import os
import sys

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

if len(sys.argv) < 3 :
    print("invalid argument ! please pass directory")
    exit()
path_img_dir = os.path.abspath(sys.argv[1])
path_face_img_dir = os.path.abspath(sys.argv[2])
list_filenames = glob.glob(path_img_dir+"/*")
index = 0
for filename in list_filenames:
    src = cv2.imread(filename)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_face = src[y:y+h,x:x+w]
        cv2.imwrite(path_face_img_dir+'/'+str(index)+'.jpg',roi_face)
        index += 1
