from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from cnn import CNN
import numpy as np


batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
size=(227, 227)
validation_split = .2
verbose = 1
num_classes = 2
patience = 100
base_path = '../models/'
data_path = '../original_faces/'


def main():
    # data generator
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    model = CNN()
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


if __name__ == '__main__':
    main()
