#!/usr/bin/env python3

from base64 import encode
from re import T
from sre_constants import SUCCESS
import numpy as np  
import cv2
import face_recognition
import os
import time
import pyttsx3

def Recognizer():

    cap = cv2.VideoCapture(0)
    engine = pyttsx3.init()
    path = "/home/goncalo/Downloads/SAVI/Parte06/Images"
    images = []
    names = []
    #variaveis para deteção e tracking
    trackers=[]
    my_list = os.listdir(path)
    for cl in my_list:
        img = cv2.imread(f"{path}/{cl}")
        images.append(img)
        names.append(os.path.splitext(cl)[0])


    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)



    print("Encoding complete")

    os.chdir(path)

    #variaveis de controlo do tempo
    tempo_de_falha = 0
    intervalo_necessario = 1
    TsinceLastDetection=0
    first_time=True
    tempo_desde_ultima_fala=0
    first_time_talking=True

    while True:
        tempo_de_Video=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        success, img_bgr = cap.read()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_small = cv2.resize(img_rgb, (0,0), None, 0.25, 0.25 )
        img_rgb_copy=img_rgb
        faces_frame = face_recognition.face_locations(img_small)
        encodes_frame = face_recognition.face_encodings(img_small, faces_frame)
        for encode_face, face_location in zip(encodes_frame, faces_frame):
            matches = face_recognition.compare_faces(encode_list, encode_face)
            face_distance = face_recognition.face_distance(encode_list, encode_face)
            match_index = np.argmin(face_distance)
            #print(face_distance)
            
            # Face detection
            if face_distance[match_index]<0.8:
                
                #Drawing Face Detections
                name = names[match_index].upper()
                y1,x2,y2,x1 = face_location
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(255,0,0),2)
                
                tracker_desconhecido=[{'track_number':'desconhecido', 'template':img_bgr[y1:y2, x1:x2], 'tempo_de_video':tempo_de_Video}]
                trackers=tracker_desconhecido

                # recognition of face from database
                if face_distance[match_index] < 0.6:
                    cv2.rectangle(img_bgr,(x1,y2+35),(x2,y2),(255,0,0),cv2.FILLED)
                    cv2.putText(img_bgr,name,(x1+6,y2+29),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    TsinceLastDetection = tempo_de_Video
                    tracker_conhecido=[{'track_number':name, 'template':img_bgr[y1:y2, x1:x2], 'tempo_de_video':tempo_de_Video}]
                    trackers=tracker_conhecido
                    first_time=True

                    # Sayibg hello the persons whose face is recognize
                    if (TsinceLastDetection-tempo_desde_ultima_fala) > 20 or first_time_talking == True:
                        engine.say('hello'+str(name))
                        engine.runAndWait()
                        tempo_desde_ultima_fala=tempo_de_Video
                        first_time_talking=False
                        
        #Tracking the face detection                 
        for tracker in trackers:
                h, w, _=tracker['template'].shape
                method=cv2.TM_CCOEFF
                result=cv2.matchTemplate(img_bgr, tracker['template'], method)
                _, _, _, max_loc= cv2.minMaxLoc(result)
                botr_x=max_loc[0]+w
                botr_y=max_loc[1]+h
                bottom_right=(botr_x, botr_y)
                tracker['template']=img_bgr[max_loc[1]:botr_y,max_loc[0]:botr_x]
                if tempo_de_Video - tracker['tempo_de_video'] > 10:
                        tracker=[]
                else:
                    cv2.rectangle(img_bgr,(max_loc),(bottom_right),(255,0,0),3)
                    cv2.putText(img_bgr,tracker['track_number'],(max_loc[0]+6,max_loc[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.imshow("webcam", img_bgr)
        if cv2.waitKey(1) == ord('q'):
            break



    cap.release()
