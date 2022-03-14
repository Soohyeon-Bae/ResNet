import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AvgPool2D, Flatten, Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from keras import backend as K
"""
# reupdated
2020.07.03.~
kangmin Park, Lab for Sensor and Modeling, Dept. of Geoinformatics, the Univ. of Seoul.
This code is utils for DCNN(Deep convolution neural network) backbones, based on Keras(Tensorflow 2.x).
updated for Semantic segmentation model.

#RESNET
def Resblock(input, knum, layer_name, pad="same", verbose=False):

    #identity mapping
    identity = input
    if verbose :
        identity = MaxPool2D(pool_size=1, strides=2)(identity)
        zero_pad = K.zeros_like(identity)
        identity = Concatenate()([identity, zero_pad])

    if not verbose :
        Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=1, padding=pad, name=layer_name+"_Conv_L1")(input)
    else :
        Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=2, padding=pad, name=layer_name+"_Conv_L1")(input)
    BN_L1 = BatchNormalization()(Conv_L1)
    AC_L1 = Activation(activation="relu")(BN_L1)

    Conv_L2 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                     strides=1, padding=pad, name=layer_name+"_Conv_L2")(AC_L1)
    BN_L2 = BatchNormalization()(Conv_L2)

    #shortcut
    shortcut = Add()([BN_L2, identity])
    shortcut = Activation(activation="relu")(shortcut)

    return shortcut

def Resblock_bn(input, knum_in, knum_out, layer_name, pad="same", verbose=False):
    """
    Residual block bottle neck version - module for over ResNet50
    :param input: input feature
    :param knum_in: the number of filters(kernels, or size of output feature) of Conv_L1 and Conv_L2
    :param knum_out: the number of filters(kernels, or size of output feature) of Conv_L3
    :param layer_name: Module name
    :param pad: "same" or "valid" ( identical with Tensorflow old version(1.x) )
    :param verbose: reduce heightxwidth(verbose=True) or not(verbose=False)
    """
    # identity mapping
    identity = input

    if not verbose : Conv_L1 = Conv2D(filters=knum_in, kernel_size=(1,1), kernel_initializer="he_normal",
                                      strides=(1,1), padding=pad, name=layer_name+"_Conv_L1")(input)
    else : Conv_L1 = Conv2D(filters=knum_in, kernel_size=(1,1), strides=(2,2), kernel_initializer="he_normal",
                            padding=pad, name=layer_name+"_Conv_L1")(input)
    BN1 = BatchNormalization()(Conv_L1)
    AC1 = Activation(activation="relu")(BN1)

    Conv_L2 = Conv2D(filters=knum_in, kernel_size=(3,3), kernel_initializer="he_normal",
                     strides=(1,1), padding=pad, name=layer_name+"_Conv_L2")(AC1)
    BN2 = BatchNormalization()(Conv_L2)
    AC2 = Activation(activation="relu")(BN2)

    Conv_L3 = Conv2D(filters=knum_out, kernel_size=(1,1), kernel_initializer="he_normal",
                     strides=(1,1), padding=pad, name=layer_name+"_Conv_L3")(AC2)
    BN3 = BatchNormalization()(Conv_L3)

    if not verbose : identity = Conv2D(filters=knum_out, kernel_size=(1,1), kernel_initializer="he_normal",
                                       strides=(1,1), padding=pad)(identity)
    else : identity = Conv2D(filters=knum_out, kernel_size=(1,1), kernel_initializer="he_normal",
                             strides=(2,2), padding=pad)(identity)

    # shortcuts
    Shortcut = Add()([BN3, identity])
    Shortcut = Activation(activation="relu")(Shortcut)

    return Shortcut

