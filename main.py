import faceutil
import numpy as np
import cv2
import socket
import sys
from threading import Thread
import traceback
import os

list_known_encoding,names = faceutil.load_data_encoding_label('./vector/*')


def recvall(sock, count):
    buf = ''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def get_tcp_img(sock):
    lenght_data = sock.recv(16)
    lenght_data = int(lenght_data.decode('ascii').rstrip())
    data_img = recvall(sock,lenght_data)
    data_img = np.fromstring(data_img,dtype='uint8')
    img = cv2.imdecode(data_img,1)
    return img

def main():
    start_server()
def start_server():
    host = "10.42.0.24"
    port = 8888
    soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    print("socket created address : "+host + ":" +str(port))
    try:
        soc.bind((host,port))
    except:
        print("Bind,failed.Error : "+str(sys.exc_info()))
        sys.exit()
    soc.listen(5)
    print("Socket now listening")
    while True:
        connection,address = soc.accept()
        ip, port = str(address[0]),str(address[1])
        print("Connected with "+ ip + ":" +port)
        try:
            Thread(target=client_thread,args=(connection,ip,port)).start()
        except:
            print("Thread did not start.")
            traceback.print_exc()
    soc.close()

def client_thread(connection,ip,port,max_buffer_size=70000):
    global list_known_encoding,names
    try:
        is_active = True
        while is_active:
            flag = connection.recv(6)
            flag = flag.decode('ascii').rstrip()
            print(flag)
            if flag == "fimg00":   
                img = get_tcp_img(connection)
                name = None
                if(len(list_known_encoding)==0):
                    name = "unknown"
                else:    
                    name = faceutil.face_recognition_ut(img,list_known_encoding,names)
                connection.sendall(name)
            if flag == "afimg0":
                numimg = connection.recv(3)
                numimg = numimg.decode('ascii').rstrip()
                numimg = int(numimg)
                connection.sendall("ok")
                lst_img = []
                for i in range(numimg):
                    img= get_tcp_img(connection)
                    lst_img.append(img)
                idnguoi = connection.recv(1024)
                idnguoi = idnguoi.decode('ascii').rstrip()
                connection.sendall("ok")
                print("saving "+str(idnguoi)+"to databse")
                faceutil.save_data_encoding(lst_img,idnguoi,'./vector')
                list_known_encoding,names = faceutil.load_data_encoding_label('./vector/*')
            if flag =="dfimg0":
                numimg = connection.recv(3)
                numimg = numimg.decode('ascii').rstrip()
                numimg = int(numimg)
                # print(numimg)
                pathvect = './vector/'+str(numimg)+'.npy'
                # print(pathvect)          
                try:
                     os.remove(pathvect)
                except:
                   print('no such file to del')
                connection.sendall("ok")
                list_known_encoding,names = faceutil.load_data_encoding_label('./vector/*')
                
                
    except:
        connection.close()
        print("disconnect from client")
main()           

