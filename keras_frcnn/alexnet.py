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
    x = Conv2D(filters=96, kernel_size=(5,5), input_shape=(32, 32, 3), activation='relu', trainable=trainable)(img_input)
    #x = keras.layers.BatchNormalization())
    x = Conv2D(filters=256, kernel_size=(5,5), activation='relu', strides=(2,2), trainable=trainable)(x)
    #x = keras.layers.BatchNormalization())
    x = Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(2,2), trainable=trainable)(x)
    x = Conv2D(filters=384, kernel_size=(3,3), activation='relu', trainable=trainable)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), activation='relu', trainable=trainable)(x)
    
    return x

def rpn(base_layers, num_anchors):

    ## Regional Proposal Network
    x = Conv2D(256, (3,3), padding='SAME', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):

    if K.backend() == 'tensorflow':
        pooling_regions = 7
        input_shape = (num_rois, 7,7,256)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 256, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
    
    

    