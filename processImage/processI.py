import os
import pickle
import pprint
import random
from PIL import Image, ImageFilter
import cv2
import PIL.ImageOps
import numpy as np
from skimage.transform import resize,warp,AffineTransform

'''
ProcessI class is used for process image, to convert it to 28*28 size from original size, and it also has
deformation method to transform its shape, but for now, we don't implement it in our train images set
'''
class ProcessI:
    def __init__(self, datart):
        self.dataroot = datart

    def processImage(self, image_name):
        '''
        hiden code should be visible when processing extra images from kaggle, it is to invert its background
        and to grey mode, and dilate it to (3,3). For other train and test or predict images, we don't need it
        '''
        # im = Image.open(self.dataroot +image_name).convert('L')
        # bkcolor = im.getpixel((0,0))
        # if bkcolor > 250:
        #     im = PIL.ImageOps.invert(im)
        # im.save(self.dataroot +image_name)
        im = cv2.imread(self.dataroot +image_name)
        im[im >= 127] = 255
        im[im < 127] = 0
        # kernel = np.ones((3,3),np.uint8)
        # im = cv2.dilate(im, kernel, iterations =1)

        image = Image.fromarray(im)

        head, tail = os.path.split(image_name)

        width = float(image.size[0])
        height = float(image.size[1])
        newImage = Image.new('L', (28, 28), (0))

        if width > height: #check which dimension is bigger
            #Width is bigger. Width becomes 20 pixels.
            nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
            if (nheight == 0): #rare case but minimum is 1 pixel
                nheight = 1
            # resize and sharpen
            img = image.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
            newImage.paste(img, (0, wtop)) #paste resized image on white canvas
        else:
            #Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((28.0/height*width),0)) #resize width according to ratio height
            if (nwidth == 0): #rare case but minimum is 1 pixel
                nwidth = 1
             # resize and sharpen
            img = image.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
            newImage.paste(img, (wleft, 0)) #paste resized image on white canvas

        # newImage.save("./annotated_28x28/"+tail, quality=100)
        tv = list(newImage.getdata())
        tva = [x * 1.0/255.0 for x in tv]
        return tva, newImage


    def image_deformation(self,image):
		random_shear_angl = np.random.random() * np.pi/6 - np.pi/12
		random_rot_angl = np.random.random() * np.pi/6 - np.pi/12 - random_shear_angl
		random_x_scale = np.random.random() * .4 + .8
		random_y_scale = np.random.random() * .4 + .8
		random_x_trans = np.random.random() * image.shape[0] / 4 - image.shape[0] / 8
		random_y_trans = np.random.random() * image.shape[1] / 4 - image.shape[1] / 8
		dx = image.shape[0]/2. \
				- random_x_scale * image.shape[0]/2 * np.cos(random_rot_angl)\
				+ random_y_scale * image.shape[1]/2 * np.sin(random_rot_angl + random_shear_angl)
		dy = image.shape[1]/2. \
				- random_x_scale * image.shape[0]/2 * np.sin(random_rot_angl)\
				- random_y_scale * image.shape[1]/2 * np.cos(random_rot_angl + random_shear_angl)
		trans_mat = AffineTransform(rotation=random_rot_angl,
									translation=(dx + random_x_trans,
												 dy + random_y_trans),
									shear = random_shear_angl,
									scale = (random_x_scale,random_y_scale))
		return warp(image,trans_mat.inverse,output_shape=image.shape)
