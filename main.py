import argparse
import pickle
import numpy as np
import cv2

from classifier.digit_classifier import HoG_LinearSVM_SingleDigitClassifier
from sign_detector import detect_sign, CurtinSignDetector

# CONFIGS
MNIST_PATH            = './data/mnist/'                # path to mnist dataset
DIGIT_CLASSIFIER_PATH = './classifier/digit_svm.yml'   # serialized digit classifier



def main(args):


    singleDigitClassifier = HoG_LinearSVM_SingleDigitClassifier()
    if args.train_digit_classifier:
        singleDigitClassifier.train_MNIST(MNIST_PATH)
        singleDigitClassifier.save(DIGIT_CLASSIFIER_PATH)
    else:
        singleDigitClassifier.load(DIGIT_CLASSIFIER_PATH)

    if args.image:
        image = cv2.imread(args.image)
        detector = CurtinSignDetector(singleDigitClassifier)
        detect_sign(image, detector)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_digit_classifier', action='store_true',
        help='Whether to train a new classifier from scratch')
    parser.add_argument('image', type=str, nargs='?', help='input image')

    args = parser.parse_args()
    main(args)
