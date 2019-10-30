import caffe
import numpy as np
import cv2
import os
import shutil


caffe_model = '/home/skliang/Downloads/gender_net.caffemodel'
deploy = '/home/skliang/Downloads/deploy_gender.prototxt'
pic_folder = '/home/skliang/workspace/data/original_faces'
output_folder = './output/pretrained'
standard_size = (227, 227)


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# 1 for female and 0 for male
def predict_gender(img_path, net):
    img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
    gray_face = cv2.resize(gray_image, standard_size)

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, 0)

    net.blobs['data'].data[...] = gray_face
    out = net.forward()
    # print (out)
    prob = out['prob']
    f_prob = prob[0][0]
    m_prob = prob[0][1]

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    prediction = 1 if f_prob > m_prob else 0
    if f_prob > m_prob:
        return 1
    else:
        return 0
    # score_list = out['global_average_pooling2d_1']

    # pre = np.argmax(score_list)

    # return pre, score_list


def main():
    net = caffe.Net(deploy, caffe_model, caffe.TEST)
    imgs = os.listdir(pic_folder)

    num = 0.0
    count = 0.0
    correct = 0.0
    for img in imgs:
        count += 1
        gender = int(img[:4])
        img_path = os.path.join(pic_folder, img)
        pred = predict_gender(img_path, net)

        if (pred == 1 and gender <= 737) or (pred == 0 and gender > 737):
            correct += 1

        print count / len(imgs)
        rate = correct / count
        print ('correct rate is ', rate)

    # rate = correct/count
    # print ('correct rate is {rate}')

if __name__ == '__main__':
    main()
