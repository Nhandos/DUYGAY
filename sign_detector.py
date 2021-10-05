import numpy as np

def pre_processing(image):

    return image


class SignDetection(object):

    def __init__(self, source_image):

        self.image = image
        self.number = None
        self.direction = None

    def __repr__(self):
        str_ = f"Number: {self.number}, Direction: {self.direction}"




def run_pipeline_on_image(image:np.ndarray, digit_classifier):


    # ---------- GLOBAL PREPROCESSING STEP ------------ #




    # ---------- REGION OF INTEREST DETECTION ---------- #


    
    
    # --------------- CLASSIFICATION --------------- #
    digit = digit_classifier.predict(image)
    print(f'Predicted number: {digit}')





