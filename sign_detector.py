from typing import Tuple, List
import numpy as np
import cv2
from classifier.digit_classifier import SingleDigitClassifier
from detector.text_detector import TextDetector


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


    def __init__(self, 
            singleDigitClassifier: SingleDigitClassifier, 
            textDetector: TextDetector 
            ):

        self.singleDigitClassifier = singleDigitClassifier
        self.textDetector = textDetector

    def detect(self, image:np.ndarray):

        scores, bboxes = self.textDetector.detect(image)

        output = image.copy()
        for x1,y1,x2,y2 in bboxes:
            cv2.rectangle(output, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
            digit, _ = self.singleDigitClassifier.predict(image[y1:y2,x1:x2])
            print(digit)

        cv2.imshow('test', output)
        cv2.waitKey(0)


def detect_sign(image:np.ndarray, sign_detector: CurtinSignDetector) -> List[CurtinSignDetection]:

    sign_detector.detect(image)
