# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K
import numpy as np
import torch

SMOOTH = 1e-6

def iou_tf(y_true, y_pred, smooth = 1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc

def iou_numpy(outputs: np.array, labels: np.array):

    intersection = (outputs & labels).sum((0,1))
    union = (outputs | labels).sum((0,1))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou  # Or thresholded.mean()    

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou    


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)