# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:53:03 2020

@author: Vedang
"""

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *



def triplet_loss(y_true, y_pred, alpha = 0.2):   
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]   
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.maximum(pos_dist - neg_dist + alpha, 0)
    loss = tf.reduce_sum(basic_loss)
    
    return loss


def LoadFRmodel():
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    
    return FRmodel

def who_is_it(image, database, model):
    encoding = EncodingImage(image, model)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc, ord='nuc')
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    cv2.imshow(identity, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity






