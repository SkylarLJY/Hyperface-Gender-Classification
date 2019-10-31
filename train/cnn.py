'''
Gender classification cnn based on the model proposed Hyperface (2016)
https://arxiv.org/pdf/1603.01249.pdf
Author: Skylar Liang
Date: Oct 30, 2019
'''
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Concatenate
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
import keras

def CNN(input_shape = (227, 227, 1)):
    model = Sequential()
    input = Input(shape=input_shape)

    conv1 = Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, activation='relu', name='conv1')(input)
    max1 = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool1')(conv1)
    max1 = BatchNormalization()(max1)

    conv1a = Conv2D(256, (4, 4), strides=(4, 4), name='conv1a', activation='relu')(max1)
    conv1a = BatchNormalization()(conv1a)

    conv2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv2', padding='same')(max1)
    max2 = MaxPooling2D((3, 3), name='max2', strides=(2, 2))(conv2)
    max2 = BatchNormalization()(max2)

    conv3 = Conv2D(384, (3, 3), strides=(1, 1) ,activation='relu', name='conv3', padding='same')(max2)
    conv3 = BatchNormalization()(conv3)

    conv3a = Conv2D(256, (2, 2), strides=(2, 2), activation='relu', name='conv3a')(conv3)
    conv3a = BatchNormalization()(conv3a)

    conv4 = Conv2D(384, (3, 3), activation='relu', name='conv4', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', name='conv5', padding='same')(conv4)
    pool5 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(conv5)
    pool5 = BatchNormalization()(pool5)

    concat = Concatenate(name='concat')([conv1a, conv3a, pool5])
    concat = BatchNormalization()(concat)

    conv_all = Conv2D(192, (1, 1), activation='relu', name='conv_all')(concat)

    flatten = Flatten()(conv_all)

    fc_all = Dense(3072, activation='relu')(flatten)

    # add more fully connected layers and out layers here if want to exploit other functions mentioned in the paper
    fc_gender = Dense(512, activation='relu')(fc_all)
    out_gender = Dense(2, activation='softmax')(fc_gender)

    model = Model(inputs=input, outputs=out_gender)

    return model


