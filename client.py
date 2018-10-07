import socket
import sys
import numpy as np
def main():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "127.0.0.1"
    port = 8888

    try:
        soc.connect((host, port))
    except:
        print("Connection error")
        sys.exit()

    print("Enter 'quit' to exit")
    message = raw_input("->")

    while message != 'quit':
        soc.sendall(message)
        rec = soc.recv(5120)
        print(rec)
        message = raw_input(" -> ")

    soc.send(b'--quit--')

# if __name__ == "__main__":
#     main()

import cv2
img = cv2.imread("./data/son/son2.png")
cv2.imshow('anh goc',img)
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
result, imgencode = cv2.imencode('.jpg', img, encode_param)
data = np.array(imgencode)
rs = cv2.imdecode(imgencode,1)
print(rs.shape)
# stringData = data.tostring()
# print(stringData)