import os
import cv2
from PIL import Image

'''
This class is used to seperate single symbol with equation according to its file name length
'''
# This class is used to seperate single and equation from the annotated data
class TestEqual:
    def getEqual(self):
        dataroot = os.getcwd() + "/data/annotated_train/"
        saveroot = os.getcwd() + "/data/train/"
        for f in os.listdir(dataroot):
            if f.endswith(".png"):
                ins = f.split('.')[0].split('_')
                if len(ins) > 3: # exclude the equation png only individual symbol
                    im = Image.open(dataroot + f)
                    im.save(saveroot + f)

def main():
    x = TestEqual()
    x.getEqual()

if __name__ == "__main__":
    main()
