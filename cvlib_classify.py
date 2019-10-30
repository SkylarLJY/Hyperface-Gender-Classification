import cvlib as cv
import cv2
import os
import numpy as np

data_path = './original_ faces'

pics = os.listdir(data_path)
for pic in pics:
    path = os.path.join(data_path, pic)
    image = cv2.imread(path)
    gender = int(pic[:4])
    print (gender)
    if gender <= 737:

    # faces, confidences = cv.detect_face(image)

    label, confidence = cv.detect_gender(image)
    print(confidence)
    print(label)

    idx = np.argmax(confidence)
    label = label[idx]

    label = "{}: {:.2f}%".format(label, confidence[idx] * 100)

    print (label)

    cv2.imshow("gender detection", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
