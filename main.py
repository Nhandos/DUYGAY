import argparse
import urllib.request
import pickle
import numpy as np
import cv2

from classifier.digit_classifier import HoG_LinearSVM_SingleDigitClassifier
from detector.text_detector import DNN_EAST_TextDetector
from sign_detector import detect_sign, CurtinSignDetector

# CONFIGS
MNIST_PATH            = './data/mnist/'                                # path to mnist dataset
DIGIT_CLASSIFIER_PATH = './classifier/digit_svm.yml'                   # digit classifier SVM model
DNN_EAST_MODEL_PATH   = './detector/frozen_east_text_detection.pb'     # text detector DNN model


def main(args):


    singleDigitClassifier = HoG_LinearSVM_SingleDigitClassifier()
    textDetector = DNN_EAST_TextDetector()

    try:
        textDetector.load(DNN_EAST_MODEL_PATH)
    except IOError:
        print('Model does not exists. Run with --download_EAST_model')
        return
    
    if args.train_digit_classifier:
        singleDigitClassifier.train_MNIST(MNIST_PATH)
        singleDigitClassifier.save(DIGIT_CLASSIFIER_PATH)
    else:
        singleDigitClassifier.load(DIGIT_CLASSIFIER_PATH)

    if args.image:
        image = cv2.imread(args.image)
        detector = CurtinSignDetector(singleDigitClassifier, textDetector)
        detect_sign(image, detector)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_EAST_model', action='store_true',
        help='Download EAST model pb file')
    parser.add_argument('--train_digit_classifier', action='store_true',
        help='Whether to train a new classifier from scratch')
    parser.add_argument('image', type=str, nargs='?', help='input image')

    args = parser.parse_args()
    main(args)
