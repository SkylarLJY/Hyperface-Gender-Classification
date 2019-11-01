import sys
import cv2
import numpy as np
import argparse
import time
import os
import shutil
from keras.models import load_model
from cnn import CNN
sys.path.append('../train')
from data_processing import preprocess_input, UTKLoad, load_data


model_path = '../models/CNN.32-0.180-0.93.hdf5'
# test_data_path = './UTKFace'
test_data_path = '../gender_data'
# test_date_path = '../positive'

if __name__ == '__main__':
    # model = CNN()
    # model.load_weights(model_path)
    model = load_model(model_path)
    print ('Done loading model')
    # test_faces, test_labels = UTKLoad(test_data_path)[:100]
    test_faces, test_labels = load_data(test_data_path)[:1000]
    print ('Done loading data')

    score = model.evaluate(test_faces, test_labels, verbose=2)
    print (score)
