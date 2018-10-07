import face_recognition
import numpy as np
import glob
import os

def save_dir_image_to_data_encoding(dirname,label,dir_path_vector):
    filenames = glob.glob(dirname)
    list_encoding = []
    for filename in filenames:
        img = face_recognition.load_image_file(filename)
        try:
            x = face_recognition.face_encodings(img)[0]
        except:
            print("file : " + filename+ " --> khong encoding duoc mat")
            continue
        list_encoding.append(x)
    list_encoding = np.asarray(list_encoding)
    filename_vec = dir_path_vector + '/'+label
    np.save(filename_vec,list_encoding)
    print("save ok")

def save_data_encoding(list_image,label,dir_path_vector):
    list_encoding = []
    for img in list_image:
        try:
            x = face_recognition.face_encodings(img)[0]
        except:
            print("can not encoding label : " +label)
            continue
        list_encoding.append(x)
    list_encoding = np.asarray(list_encoding)
    filename_vec = dir_path_vector + '/'+label
    np.save(filename_vec,list_encoding)
    print("save ok")


def load_data_encoding_label(dir_path):
    filenames = glob.glob(dir_path)
    list_encoding = []
    label = []
    for filename in filenames:
        x = np.load(filename)
        if x.ndim == 2:
            for i in range(x.shape[0]):
                list_encoding.append(x[i,:])
                label.append(os.path.basename(filename).split('.')[0])
        else:
            list_encoding.append(x)
            label.append(os.path.basename(filename).split('.')[0])
        
    list_encoding = np.asarray(list_encoding)
    return (list_encoding,label)

def face_recognition_ut_v2(image,list_encoding,label,x,y):
    image_encoding = face_recognition.face_encodings(image)
    if np.shape(image_encoding)[0] == 0:
        return x,y, 'unknown'
    image_encoding = np.asarray(image_encoding[0])
    # print(np.shape(list_encoding-image_encoding))
    scores = np.linalg.norm(list_encoding-image_encoding,axis=1)
    index = np.argmin(scores)

    if scores[index] >= 0.5 :
        return x,y, 'unknown'
    else :
        return x,y, label[index]
def face_recognition_ut(image,list_encoding,label):

    image_encoding = face_recognition.face_encodings(image)
    if np.shape(image_encoding)[0] == 0:
        return 'unknown'
    image_encoding = np.asarray(image_encoding[0])
    scores = np.linalg.norm(list_encoding-image_encoding,axis=1)
    index = np.argmin(scores)

    if scores[index] >= 0.5 :
        return 'unknown'
    else :
        return label[index]

    
    
    