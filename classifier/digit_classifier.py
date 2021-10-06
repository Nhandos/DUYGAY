from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Iterable, List

import cv2
import numpy as np
import tqdm
from mnist import MNIST


class DigitClass(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


class SingleDigitClassifier(ABC):

    def __init__(self):
            pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[DigitClass, float]:
        """ returns prediction """

    @abstractmethod
    def load(self, model_path: str):
        """ load classifier"""

    @abstractmethod
    def save(self, model_path: str):
        """ save classifier"""


class HoG_LinearSVM_SingleDigitClassifier(SingleDigitClassifier):

    def __init__(self):

        # ---------- SVM Classifier ---------- #
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        
        # ---------- HoG Extractor ---------- #
        self.winSize = (56, 56)
        self.blockSize = (self.winSize[0] // 2, self.winSize[1] // 2)
        self.blockStride = (self.winSize[0] // 8, self.winSize[1] // 2)
        self.cellSize = (self.winSize[0] // 14, self.winSize[0] // 14)
        self.nBins = 8

    def _preprocess(self, image):
        image = cv2.resize(image, self.winSize)
        return image

    def _extract_features(self, image:np.ndarray):

        image = self._preprocess(image)
        hog = cv2.HOGDescriptor(
            self.winSize,
            self.blockSize,
            self.blockStride,
            self.cellSize,
            self.nBins)
                 
        feature_vector = hog.compute(image)

        return feature_vector.flatten().astype(np.float32)

    def save(self, model_path: str):
        """ Save model to file """
        self.svm.save(model_path)

    def load(self, model_path: str):
        """ Load model from file """
        self.svm = self.svm.load(model_path)

    def predict(self, image: np.ndarray) -> Tuple[DigitClass, float]:
        """ Predicts digit class and a confidence score """

        feature_vector = self._extract_features(image)
        _, response = self.svm.predict(np.array([feature_vector], dtype=np.float32))

        return response.squeeze(), 1.0

    def batch_predict(self, images: Iterable[np.ndarray]) -> List[Tuple[DigitClass, float]]:
        """ Predict digit class and confidence score on multiple images """

        features = []
        for image in tqdm.tqdm(images):
            features.append(self._extract_features(image))

        features = np.array(features).astype(np.float32)
        _, responses = self.svm.predict(features)
        result = [(response.squeeze(), 1.0) for response in responses]

        return result

    def train_MNIST(self, dataset_path):
        """ Train the classifier with MNIST dataset """

        mndata = MNIST(dataset_path)
        train_images, train_labels = mndata.load_training()
        test_images, test_labels  = mndata.load_testing()
        
        # Convert to numpy images
        train_images = [np.array(image, dtype=np.uint8).reshape(28, 28) for
                image in train_images]
        test_images = [np.array(image, dtype=np.uint8).reshape(28, 28) for
                image in test_images]

        print('Extracting Feature from images')
        X = []
        for image in tqdm.tqdm(train_images):
            X.append(self._extract_features(image).flatten())
        X = np.array(X)
        Y = np.array(train_labels, dtype=int)

        print('Training with training set')
        self.svm.train(X, cv2.ml.ROW_SAMPLE, Y)
        X_pred = [label for label, score in self.batch_predict(train_images)]
        n_error = sum([1 if expected != actual else 0 for expected, actual in zip(Y, X_pred)])
        print(f'Error rate = {n_error/len(train_images) * 100}%') 
        
        print('Testing on validation set')
        X = self.batch_predict(test_images)
        Y = test_labels
        n_error = sum([1 if expected != actual[0] else 0 for expected, actual in zip(Y, X)])
        print(f'Error rate = {n_error/len(test_images) * 100}%') 

