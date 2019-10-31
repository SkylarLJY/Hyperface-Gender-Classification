from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from cnn import CNN
from data_processing import load_data, preprocess_input, split_data
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
data_path = '../gender_data/'


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

    # callbacks
    f = open(base_path + 'gender_classification_training.log', 'w')
    f.close()
    log_file_path = base_path + 'gender_classification_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)

    trained_models = base_path + 'CNN.{epoch:02d}-{val_loss:.3f}-{val_accuracy:.2f}.hdf5'
    model_cp = ModelCheckpoint(trained_models, 'val_loss', verbose=1, save_best_only=True)
    callbacks = [model_cp, csv_logger, early_stop, reduce_lr]

    # load data
    faces, labels = load_data(data_path)
    print (len(faces))
    print (len(labels))
    faces = preprocess_input(faces)
    order = np.argsort(np.random.random(len(faces)))
    faces = faces[order]
    labels = labels[order]

    train_data, val_data = split_data(faces, labels, validation_split)
    train_faces, train_labels = train_data
    model.fit_generator(data_generator.flow(train_faces, train_labels, batch_size),
                        steps_per_epoch=len(train_faces)/batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_data)



if __name__ == '__main__':
    main()
