from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import cv2

from detector.utils import non_max_suppression

class ModelNotLoadedError(Exception):
    pass

class TextDetector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def save(self, model_path: str):
        """ Save model to file """

    @abstractmethod
    def load(self, model_path: str):
        """ Load model from file """

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[Tuple]]:
        """ Detect text in image """


class DNN_EAST_TextDetector(TextDetector):

    def __init__(self):

        self.EASTLayers = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]
        self.min_confidence = 0.8
        self.W = 320
        self.H = 320
        self.dnn = None

    def load(self, model_path: str):
        """ Load model from file """
        self.dnn = cv2.dnn.readNet(model_path)

    def save(self, model_path: str):
        """ Save model to a file """
        raise NotImplementedError('Saving this model not supported')

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple]]:
        """ Detects Text regions in single image 

            For each detect text regions there will be a two-value
            tuple with the bounding-box and confidence score

                bounding-box: Tuple[x, y, w, h]
                score: (float) between [0, 1]

            Args:
                image: (np.narray) input image 

            Returns:
                List[Tuple[score, bounding-box]]
        """

        if self.dnn == None:
            raise ModelNotLoadedError("The model has not been loaded")

        height, width = image.shape[:2]
        rW = width / self.W
        rH = height / self.H

        image = cv2.resize(image, (self.W, self.H))
        blob = cv2.dnn.blobFromImage(image, 
                scalefactor=1.0, 
                size=(self.W, self.H),
                mean=(123.6, 116.78, 103.94),
                swapRB=True,
                crop=False,
                ddepth=cv2.CV_32F
            )

        self.dnn.setInput(blob)
        (scores, geometry) = self.dnn.forward(self.EASTLayers)

        rects, confidences = self._decode_predictions(scores, geometry)
        keep = non_max_suppression(np.array(rects), confidences, 0.1)
        confidences = confidences[keep]
        absolute_rects = []
        for x1,y1,x2,y2 in rects[keep]:
            x1 = int(x1 * rW)
            y1 = int(y1 * rH)
            x2 = int(x2 * rW)
            y2 = int(y2 * rH)
            absolute_rects.append((x1,y1,x2,y2))


        return confidences,absolute_rects 

    def _decode_predictions(self, scores, geometry):
        """ ref: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector"""

        (numRows, numCols) = scores.shape[2:4]	
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < self.min_confidence:
                    continue
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        # return a tuple of the bounding boxes and associated confidences
        return (np.array(rects), np.array(confidences))




             




            
