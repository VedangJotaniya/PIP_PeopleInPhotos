# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:53:52 2020

@author: Vedang
"""

import cv2
import os
import json
import csv
import numpy as np
from FaceRecognition        import *
from FaceDetection          import *
from fr_utils               import *
from inception_blocks_v2 import *

FaceCascade = None
db = {}
FRmodel =None
def main():
    FaceCascade = cv2.CascadeClassifier('.\\include\\haarcascade_frontalface_default.xml')
    with open('.\\SavedEncodings\\KnownPersons..csv', 'r') as openfile:
        reader = csv.reader(openfile)
        db = dict(reader)
        
    FRmodel = LoadFRmodel()
    print("Main Funcion Executed")  
    return FaceCascade, db, FRmodel
        
def SaveProfile(Name, Id, img, db):
    encoding = EncodingImage(img, FRmodel)
    db[Id] = encoding
    return db

def SaveDB(db, File):
    # with open(File, 'w') as writefile:
    #     json.dump(db, writefile)
    with open(File, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in db.items():
           writer.writerow([key, value])
    return 

def FindPersons(path, db, FRmodel):
    
    
    
    
    return



if __name__ == "__main__":
    FaceCascade, db, FRmodel = main()
    print("Main is executed")



Name = "Krishna"
Id = "022"
img = cv2.imread(".\\SavedEncodings\\images\\022.jpg")
db = SaveProfile(Name, Id, img, db) 
SaveDB(db, '.\\SavedEncodings\\KnownPersons.csv')



    