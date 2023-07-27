import csv

import pandas as pd;
import matplotlib.pyplot as plt;
import tensorflow as tf;
from tensorflow import keras ;
import numpy as np;

from Utils.model_utils import early_stopping_binary,learning_rate_reduction_binary
from Utils.model_utils import Precision,Recall,F1




train_data_x,train_data_y = Utils.data_utils.KDD_Train_data()

"""Select the test data for testing, test data include 'KDD Test+' and 'KDD Test-21' """
#test_data_x,test_data_y = Utils.data_utils.KDD_Test_data(Utils.data_utils.kdd_test)
test_data_x,test_data_y = Utils.data_utils.KDD_Test_data(Utils.data_utils.kdd_test_21)

model = keras.Sequential(
    [keras.layers.GRU(128,input_shape=(1,4)),
     keras.layers.LeakyReLU(alpha=0.5),
     keras.layers.Dense(32),
     keras.layers.LeakyReLU(alpha=0.5),
     keras.layers.Dense(16),
     keras.layers.LeakyReLU(alpha=0.5),
     keras.layers.Dense(1,activation='sigmoid')
     ]
);


model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['binary_accuracy',
              #Precision,
	      	  #Recall,
	      	  F1]
              );

history = model.fit(train_data_x,train_data_y,epochs=300,batch_size=512,validation_data=(test_data_x, test_data_y),shuffle = True, callbacks=[early_stopping_binary,learning_rate_reduction_binary]);

test_history = model.evaluate(test_data_x,test_data_y);







