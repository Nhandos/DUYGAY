import glob
import os
import unittest
import cv2
from classifier.digit_classifier import HoG_LinearSVM_SingleDigitClassifier

MODEL_PATHS = {
    'HoG_LinearSVM_SingleDigitClassifier': './classifier/digit_svm.yml'
}

DIGIT_TEST_IMAGE_PATH = './data/digits_original'


class Test_HoG_LinearSVM_SingleDigitClassifier(unittest.TestCase):

    def setUp(self):
        self.digit_classifier = HoG_LinearSVM_SingleDigitClassifier()
        self.digit_classifier.load(MODEL_PATHS[self.__class__.__name__[5:]])

    def test_error_rate(self):
        
        n_images = 0
        total_errors= 0
        for i, digit in enumerate(['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']):
            
            print('Testing class: ', digit)
            img_paths = glob.glob(os.path.join(DIGIT_TEST_IMAGE_PATH, f'./{digit}*.jpg'))
            images = []
            for img_path in img_paths:
                images.append(cv2.imread(img_path))

            predictions = self.digit_classifier.batch_predict(images)
            n_errors = 0
            for label, _ in predictions:
                if label != i:
                    n_errors += 1
            print(f'    Error rate = {n_errors/len(images) * 100}%')
            n_images += len(images)
            total_errors += n_errors
        print()
        print(f'Total Error rate = {total_errors/n_images * 100}%')

