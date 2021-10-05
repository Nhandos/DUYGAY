import argparse
import pickle
from mnist import MNIST
import numpy as np
import cv2

from classifier.digit_classifier import DigitClassifier
from sign_detector import run_pipeline_on_image

# CONFIGS
MNIST_PATH            = './data/mnist/'                # path to mnist dataset
DIGIT_CLASSIFIER_PATH = './classifier/digit_svm.ser'   # serialized digit classifier

def main(args):

    if args.train_digit_classifier:
        # Load training data
        print('Loading training data')
        mndata = MNIST(MNIST_PATH)
        train_images, train_labels = mndata.load_training()
        test_images, test_labels  = mndata.load_testing()
        train_images = [np.array(image, dtype=np.uint8).reshape(28, 28) for
                image in train_images]
        test_images = [np.array(image, dtype=np.uint8).reshape(28, 28) for
                image in test_images]

        # Training new classifier
        print('Training new classifier')
        digit_classifier = DigitClassifier()
        digit_classifier.train(train_images, train_labels, test_images, test_labels)
        with open(DIGIT_CLASSIFIER_PATH, "wb") as fp:
            pickle.dump(digit_classifier, fp)

    digit_classifier = None
    try:
        with open(DIGIT_CLASSIFIER_PATH, "rb") as fp:
            digit_classifier = pickle.load(fp)

        if digit_classifier is None:
            print("Could not load digit classifier")
            return

    except (OSError, IOError) as e:
        print("Digit Classifier does not exist, run with --train_digit_classifier to train new classifier")
        return

    image = cv2.imread(args.image)
    run_pipeline_on_image(image, digit_classifier)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_digit_classifier', action='store_true',
        help='Whether to train a new classifier from scratch')
    parser.add_argument('image', type=str, help='input image')

    args = parser.parse_args()
    main(args)
