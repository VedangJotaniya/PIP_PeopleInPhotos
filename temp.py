# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:36:14 2020

@author: Vedang
"""

from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
from FaceRecognition import *
from fr_utils import *
from main import *
import cv2

def who_is_it1(image, database, model):
    encoding = model.predict(np.array([image]))
    min_dist = 100
    identity = "None Found"
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc, ord='nuc')
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    
    cv2.imshow("Amitabh", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return min_dist, identity

def FindPersons1(path, db, FRmodel):
    PersonsFound={}
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG)$", filename):
                # filepath = os.path.join(root, filename)
                image = cv2.imread(path + filename)
                dist, ID = who_is_it1(image, db, FRmodel)
                print(str(dist) + " " + str(ID))
                PersonsFound[path+filename] = ID 
    
    return PersonsFound

with CustomObjectScope({'tf': tf}):
    model = load_model('.\\model\\nn4.small2.channel_first.h5')
    src_img = cv2.imread('.\\SavedEncodings\\images\\002.jpg')
    
    print(src_img.shape)
    
    db = {}
    
    input_x = np.array([src_img])
    
    db["Amitabh"] = model.predict(np.array([src_img]))
    
    result = FindPersons1(".\\GeneratedImage\\2020-06-26\\c\\", db, model)