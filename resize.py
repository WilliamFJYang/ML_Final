import numpy as np 
import cv2
import os



for fold_cnt in range (1 , 5):
    file_count = sum(len(files) for _, _, files in os.walk("../input/" + str(fold_cnt)))
    for cnt in range (1,file_count+1):
        img = cv2.imread("../input/" + str(fold_cnt) + "/" + str(fold_cnt) + "(" + str(cnt) + ").jpg") # BRG order, uint8 type
        print("../input/" + str(fold_cnt) + "/" + str(fold_cnt) + " _ (" + str(cnt) + ").jpg")
        rows,cols,a = img.shape
        if rows > cols :
            img = cv2.resize(img,(720,int(rows*720/cols)))
        else:
            img = cv2.resize(img,(int(cols*720/rows),720))
        cv2.imwrite("H:/resize_temp/" + str(fold_cnt) + "/" + str(cnt) + ".jpg",img)

