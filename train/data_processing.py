import pandas as pd
import numpy as np
import os
import cv2


def load_data(path, size=(227, 227)):
    faces = []
    labels = []
    file_list = ['male', 'female']
    label_list = [0, 1]

    for i in range(0, len(file_list)):
        file_path = os.path.join(path, file_list[i])
        img_list = os.listdir(file_path)
        for img in img_list:
            img_path = os.path.join(file_path, img)
            im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, size)
            faces.append(im.astype('float32'))
            labels.append(label_list[i])

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    labels = pd.get_dummies(labels).as_matrix()

    return faces, labels


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


def UTKLoad(path, size=(227, 227)):
    imgs = os.listdir(path)
    faces = []
    labels = []
    file_list = ['male', 'female']
    label_list = [0, 1]
    for i in imgs:
        age, gender, _ = i.split('_', 2)
        age = int(age)
        gender = int(gender)
        gender = file_list[gender]
        # babies and senior people don't have distinctive gender features so not selected
        # if age < 15 or age > 70:
        #     continue
        face = cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE)
        face = cv2.resize(face, size)
        faces.append(face.astype('float32'))
        labels.append(gender)
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    labels = pd.get_dummies(labels).as_matrix()

    return faces, labels

