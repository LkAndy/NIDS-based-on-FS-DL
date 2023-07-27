import csv

import pandas as pd;
import matplotlib.pyplot as plt;
import tensorflow as tf;
from tensorflow import keras ;
import numpy as np;
from sklearn.metrics import classification_report

from Utils.model_utils import early_stopping_multi,learning_rate_reduction_multi
from Utils.model_utils import Precision,Recall,F1

import Utils.data_utils



train_data_x,train_data_y = Utils.data_utils.KDD_Train_data()


"""Select the test data for testing, test data include 'KDD Test+' and 'KDD Test-21' """
#test_data_x,test_data_y = Utils.data_utils.KDD_Test_data(Utils.data_utils.kdd_test)
test_data_x,test_data_y = Utils.data_utils.KDD_Test_data(Utils.data_utils.kdd_test_21)

# Multi classfication
train_data_y = keras.utils.to_categorical(train_data_y, num_classes=5)
test_data_y = keras.utils.to_categorical(test_data_y, num_classes=5)


model = keras.Sequential(
    [keras.layers.GRU(128,input_shape=(1,4)),
     keras.layers.LeakyReLU(alpha=0.5),
     keras.layers.Dense(32),
     keras.layers.LeakyReLU(alpha=0.5),
     keras.layers.Dense(16),
     keras.layers.LeakyReLU(alpha=0.5),
     keras.layers.Dense(5,activation='softmax')
     ]
);

model.compile(optimizer='adamax',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy',
              #Precision,
	      	  #Recall,
	      	  F1]
              );
history = model.fit(train_data_x,train_data_y,epochs=300,batch_size=512,validation_data=(test_data_x, test_data_y),shuffle = True, callbacks=[early_stopping_multi,learning_rate_reduction_multi]);

test_history = model.evaluate(test_data_x,test_data_y);








