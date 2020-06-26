# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:53:52 2020

@author: Vedang
"""

import opencv as cv2
import os
import json
from FaceRecognition        import *
from FaceDetection          import *
from fr_utils               import *
from inception_blocks_v2.py import *

FaceCascade = None
db = {}
def main():
    FaceCascade = cv2.CascadeClassifier('.\\include\\haarcascade_frontalface_default.xml')
    with open('.\\SavedEncodings\\KnownPersons.json', 'r') as openfile:
        db = json.load(openfile) 
    print("Main Funcion Executed")  
    return 
        
def SaveProfile(Name, ):
    
    
    return



if __name__ == "__main__":
    main()


