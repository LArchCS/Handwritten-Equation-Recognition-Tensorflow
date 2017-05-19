import os
import pickle
import pprint
import random
from PIL import Image, ImageFilter
import cv2
import numpy as np
from processI import ProcessI
from skimage.transform import resize,warp,AffineTransform

'''
Get the all .png files in the train or test directory, and using the ProcessImage method in ProcessI class
to process the image, and save it in the same place, thus we can get 28*28 grey mode image for both train and test images of
single symbols
'''

class PreSymbol:
    def __init__(self):
        self.dataroot = os.getcwd() + "/data/train/" # change to "/data/test/"
        self.ps = ProcessI(self.dataroot)

    def getsymbol(self):

        for f in os.listdir(self.dataroot):
            if f.endswith(".png"):
                im1,im2 = self.ps.processImage(f) #im1 is image data, im2 is image
                im2.save(self.dataroot+f) #save 28*28 size image 

if __name__ == "__main__":
    PreSymbol().getsymbol()
