from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

early_stopping_binary =EarlyStopping(monitor='val_binary_accuracy',min_delta=0.1, patience=100, restore_best_weights = True, mode='max')

learning_rate_reduction_binary = ReduceLROnPlateau(monitor='val_binary_accuracy', patience=20, factor=0.5, min_lr=0.00001)

early_stopping_multi =EarlyStopping(monitor='val_categorical_accuracy',min_delta=0.1, patience=100, restore_best_weights = True, mode='max')

learning_rate_reduction_multi = ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=20, factor=0.5, min_lr=0.00001)



from keras import backend as K
import pandas as pd;

def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1