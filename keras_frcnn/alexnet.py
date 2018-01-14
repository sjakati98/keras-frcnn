''' AlexNet model for smaller training times'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


def nn_base(input_tensor=None, trainable=False):


    ## Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3,None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    else:
        img_input = Input(shape=input_shape)

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1


    ## Conv Block
    x = Conv2D(filters=48, kernel_size=(5,5), input_shape=(32, 32, 3), activation=relu, data_format='channels_last')(img_input)
    #x = keras.layers.BatchNormalization())
    x = Conv2D(filters=256, kernel_size=(5,5), activation='relu', strides=(2,2))(x)
    #x = keras.layers.BatchNormalization())
    x = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(2,2))(x)
    x = Conv2D(filters=384, kernel_size=(3,3), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), activation='relu')(x)
    
    ## TODO: figure out whether or not the dense layers go here or not lol
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    
    
    

    