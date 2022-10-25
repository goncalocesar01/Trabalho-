#!/usr/bin/env python3


import numpy as np  
import cv2
import face_recognition
import os
import time
from copy import deepcopy


def Detector():
    path = "/home/goncalo/Downloads/SAVI/Parte06/Images"
    face_cascade = cv2.CascadeClassifier("/home/goncalo/Downloads/SAVI/Parte06/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    os.chdir(path)

    t1 = time.time()

    while (cap.isOpened()):
    
        ret, img_bgr = cap.read()
        image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_copy = deepcopy(img_bgr)
        faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.5, minNeighbors=5)
    
        
        for (x,y,w,h) in faces:
            t = time.time()
            roi = img_copy[y-50:(y+h)+50, x-50:(x+w)+50]
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(img_bgr, (x,y), (x+w,y+h), color, stroke)
            
            if (t - t1) > 5:
                cv2.putText(img_bgr,"Who are you?",(x,(y+h)+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                print("Press p to take a photo")
                
                if  cv2.waitKey(50) == ord('p'):
                    print("Type your name")
                    name = input()
                    name_input = name + ".jpg"
                    cv2.imwrite(name_input, roi)
                    t1 = time.time()
                    
            

                
        cv2.imshow("webcam", img_bgr) 
        if cv2.waitKey(50) == ord('q'):
            break

    cap.release()
    



