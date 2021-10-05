import argparse
import cv2
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
from matplotlib import pyplot as plt
import matplotlib
import tqdm
from sklearn import svm
matplotlib.use('TkAgg')

class DigitClassifier(object):


    def __init__(self):

        self.svm = None
        self.lin_clf = svm.LinearSVC()

        self.image_width = 28 
        self.image_height = 28
        self.cells_per_block= (self.image_height // 2, self.image_width // 2)
        self.n_orient = 8
        self.pixels_per_cell = (2, 2)

        self.n_blocks_row = self.image_height // self.pixels_per_cell[0]
        self.n_blocks_col = self.image_width // self.pixels_per_cell[1]
        self.block_stride = (self.image_height // 4, self.image_width // 4)
        self.n_cells_row = self.cells_per_block[0]
        self.n_cells_col = self.cells_per_block[1]

    def _preprocess(self, image):
        image = cv2.resize(image, dsize=(self.image_width, self.image_height))
        return image 

    def extract_features(self, image:np.ndarray):
        assert image.shape[:2] == (self.image_height, self.image_width)

        #fd = hog(image, orientations=self.n_orient, pixels_per_cell=self.pixels_per_cell,
        #        cells_per_block=self.cells_per_block, visualize=False, feature_vector=False)
        #fd = image 
        hog = cv2.HOGDescriptor((self.image_height, self.image_width),
                self.cells_per_block, self.block_stride, self.pixels_per_cell,
                self.n_orient, 1, -1.0, 0, 0.2, 1, 64, True)
        fd = hog.compute(image)

        #assert fd.shape == self.feature_shape 

        return fd.flatten()

    def predict(self, image):
        image = self._preprocess(image)
        features = self.extract_features(image)
        return self.lin_clf.predict(features.reshape(1, -1))

    def train(self, train_images, train_labels, test_images, test_labels):

        X = []
        Y = []

        n_samples = len(train_images)

        print('Extracting Feature from images')
        for (image, label) in tqdm.tqdm(list(zip(train_images, train_labels))):
            image = self._preprocess(image)
            fd = self.extract_features(image)
            X.append(fd.flatten())
            Y.append(label)

        Y = np.array(Y).reshape(n_samples,)
        X = np.array(X).reshape(n_samples, -1)

        print('Fitting SVM')
        self.lin_clf.fit(X, Y)

        # Testing
        n_errors =  0
        print('Testing SVM')
        for (image_path, label) in tqdm.tqdm(list(zip(test_images, test_labels))):

            # pre-processing
            image = self._preprocess(image)
            fd = self.extract_features(image)

            fd = fd.reshape(1, -1)
            predict = self.lin_clf.predict(fd)
            n_errors = n_errors + 1 if predict != label else n_errors

        print('Error rate = {}% ({}/{})'.format(100 - n_errors/len(test_images) * 100,
            n_errors, len(test_images)))


