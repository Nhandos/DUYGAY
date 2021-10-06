import numpy as np
from typing import Tuple, List
from classifier.digit_classifier import SingleDigitClassifier

def pre_processing(image):

    return image


class CurtinSignDetection(object):

    def __init__(self, image: np.ndarray, number: str, direction: str, roi:
        Tuple[int, int, int, int]):

        self.image = image
        self.number = number
        self.direction = direction
        self.roi = roi

    def get_visualisation(self) -> np.ndarray:
        x, y, w, h = self.roi
        return self.image[y:y+h,x:x+w,:]

    def __repr__(self):
        str_ = f"Number: {self.number}, Direction: {self.direction}"


class CurtinSignDetector(object):


    def __init__(self, singleDigitClassifier: SingleDigitClassifier):
        self.singleDigitClassifier = singleDigitClassifier

    def detect(self, image:np.ndarray):

        digit, _ = self.singleDigitClassifier.predict(image)
        print(digit)


def detect_sign(image:np.ndarray, sign_detector: CurtinSignDetector) -> List[CurtinSignDetection]:

    sign_detector.detect(image)
