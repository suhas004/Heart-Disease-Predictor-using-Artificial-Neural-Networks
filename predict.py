
"""
Created on sat oct 20:45:05 2020

@author: suhas
"""

import numpy as np
from tensorflow.keras.models import load_model


class heart_disease_predictor:
    def __init__(self,attributes):
        self.attributes =attributes


    def prediction(self):
        # load model
        model = load_model('heart.h5')

        #model.summary()
        attributes = self.attributes

        result = model.predict(attributes)
        print(result)
        if result > 0.45:
            prediction = 'Severe'
            print(prediction)
        else:
            prediction = 'Normal'
            print(prediction)

val=[[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]



a=heart_disease_predictor(val)
a.prediction()