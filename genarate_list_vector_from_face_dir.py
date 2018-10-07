import cv2
import glob
import faceutil
import os
import sys

if len(sys.argv) < 3 :
    print("invalid argument ! please pass directory")
    exit()

path_dir_face_img = os.path.abspath(sys.argv[1])
path_dir_vector = os.path.abspath(sys.argv[2])

# print(glob.glob(path_dir_face_img+'/*'))

# list_name = [os.path.basename(x) for x in glob.glob(path_dir_face_img+'/*')]

for path_face_name in glob.glob(path_dir_face_img+'/*'):
    name_class = os.path.basename(path_face_name)
    faceutil.save_dir_image_to_data_encoding(path_face_name+'/*',name_class,path_dir_vector)
