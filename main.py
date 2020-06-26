# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:53:52 2020

@author: Vedang
"""

import cv2
import os
import json
import csv
import re
import numpy as np
from FaceRecognition        import *
from FaceDetection          import *
from fr_utils               import *
from inception_blocks_v2 import *

       
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
    PersonsFound={}
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG)$", filename):
                # filepath = os.path.join(root, filename)
                image = cv2.imread(path + filename)
                dist, ID = who_is_it(image, db, FRmodel)
                print(str(dist) + " " + str(ID))
                PersonsFound[path+filename] = ID 
    
    return PersonsFound



if __name__ == "__main__":
    print("Main Started Executing")
    FaceCascade = cv2.CascadeClassifier('.\\include\\haarcascade_frontalface_default.xml')
    print("Loading HaarCascade Classifier")
    # with open(".\\SavedEncodings\\KnownPersons.json") as readfile:
    print("Loading encoding model")
    FRmodel = LoadFRmodel()
    print("model loaded")
    
    imgname = "6"
    img = cv2.imread(".\\images\\processing\\" + imgname + ".jpg")
    ScaleFactor = 1.1
    MinNeighbours = 4
    SavePath = ".\\GeneratedImage\\2020-06-26\\c\\"
    __ = GenerateFaces(imgname, img, FaceCascade, ScaleFactor, MinNeighbours, SavePath)
    
    db={}
    ID = "004"
    img = cv2.imread(".\\SavedEncodings\\images\\"+ID+".jpg")
    db[ID] = EncodingImage(img, FRmodel)

    personsPresent = FindPersons(".\\GeneratedImage\\2020-06-26\\b\\", db, FRmodel)

# Name = "Krishna"
# Id = "022"
# img = cv2.imread(".\\SavedEncodings\\images\\022.jpg")
# db = SaveProfile(Name, Id, img, db) 
# SaveDB(db, '.\\SavedEncodings\\KnownPersons.csv')




 