# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:53:03 2020

@author: Vedang
"""

import cv2
import datetime as dt

def DetectStructure(image, FaceCascade, ScaleFactor, MinNeighbours):    
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    FaceCoordinates = FaceCascade.detectMultiScale(grayScale, ScaleFactor, MinNeighbours)    

    return FaceCoordinates

def SaveCroppedImage(ImgName, image, path, shape=(3, 96, 96)):
    image =  cv2.resize(image, (96, 96))
    cv2.imwrite(path+ImgName, image)
    

def GenerateFaces(imgname, img, ScaleFactor, MinNeighbours, SavePath):
    img=cv2.resize(img, (720, int(720 * img.shape[0] / img.shape[1])))
    
    FaceCascade = cv2.CascadeClassifier('.\\include\\haarcascade_frontalface_default.xml')
    Faces = DetectStructure(img, FaceCascade, ScaleFactor, MinNeighbours)
    
    Today = dt.datetime.now()
    DateStamp = str(Today.year) + "-" + str(Today.month) + "-" + str(Today.day) 
    
    i=0
    for (x, y, w, h) in Faces:
        SaveName = DateStamp + "_" + str(i) + "_" + imgname + ".jpg"
        SaveCroppedImage(SaveName, img[y:y+h, x:x+w], SavePath)
        i+=1
                        
    return True 

            # imgname="A221"
            # img = cv2.imread(".\\images\\processing\\16.jpg")
            # ScaleFactor = 1.1
            # MinNeighbours = 1
            # SavePath = ".\\GeneratedImage\\2020-06-26\\A221\\"
            # status = GenerateFaces(imgname, img, ScaleFactor, MinNeighbours, SavePath)
            # print(status)

