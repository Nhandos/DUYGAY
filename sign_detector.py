from typing import Tuple, List
import numpy as np
import cv2
from classifier.digit_classifier import SingleDigitClassifier
from detector import utils
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
            textDetector: TextDetector,
            regionDetector: cv2.MSER
            ):

        self.singleDigitClassifier = singleDigitClassifier
        self.textDetector = textDetector
        self.mser = regionDetector

    def detect(self, image:np.ndarray):

        image = cv2.GaussianBlur(image, (3,3), 1.6)

        scores, bboxes = self.textDetector.detect(image)
        output = image.copy()

        darkest_background_value = 255.0
        darkest_idx = 0
        for i, (x1,y1,x2,y2) in enumerate(bboxes):

            cropped = image[y1:y2,x1:x2,:]
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            background_value = np.average(hsv[...,2])
    
            if background_value < darkest_background_value:
                darkest_idx = i
                darkest_background_value = background_value


        x1, y1, x2, y2 = bboxes[darkest_idx]
        x1 = max(x1 - 5, 0)
        y1 = max(y1 - 5, 0)
        x2 = min(x2 + 5, image.shape[1])
        y2 = min(y2 + 5, image.shape[0])
        hsv = cv2.cvtColor(image[y1:y2,x1:x2,...], cv2.COLOR_BGR2HSV)
        self.mser.setMaxArea(int((x2 - x1) * (y2 - y1) * 0.1))
        self.mser.setMinArea(int((x2 - x1) * (y2 - y1) * 0.05))
        regions, _ = self.mser.detectRegions(hsv[:,:,2])

        if len(regions) > 0:
            digit_bboxes = [] 
            for p in regions:
                xmin, ymin = np.amin(p,axis=0)
                xmax, ymax = np.amax(p,axis=0)
                digit_bboxes.append((xmin,ymin,xmax,ymax))

            digit_bboxes = np.array(digit_bboxes)
            areas = (digit_bboxes[:,3] - digit_bboxes[:,1]) * (digit_bboxes[:,2] -
                    digit_bboxes[:,0])
            keep_idx = utils.non_max_suppression(digit_bboxes, areas, 0.05)

            number_location = []
            for i in range(digit_bboxes.shape[0]):
                if i in keep_idx:
                    xmin, ymin, xmax, ymax = digit_bboxes[i, ...]

                    # use relative constants?
                    xmin = max(xmin - 5, 0)
                    ymin = max(ymin - 5, 0)
                    xmax = min(xmax + 5, hsv.shape[1])
                    ymax = min(ymax + 5, hsv.shape[0])

                    cv2.rectangle(output, (x1+xmin, y1+ymax), (x1+xmax, y1+ymin), color=(0,255,0), thickness=1)
                    digit, _ = self.singleDigitClassifier.predict(hsv[ymin:ymax,xmin:xmax,2])
                    number_location.append((xmin, str(int(digit))))
            number_location.sort(key=lambda x: x[0])
            print(''.join([number[1] for number in number_location]))

        # non-max suppression
        cv2.rectangle(output, (x1,y1), (x2,y2), color=(255,0,0), thickness=3)
        cv2.imshow('test', output)
        cv2.waitKey(0)


def detect_sign(image:np.ndarray, sign_detector: CurtinSignDetector) -> List[CurtinSignDetection]:

    sign_detector.detect(image)
