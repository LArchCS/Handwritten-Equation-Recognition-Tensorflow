import pickle
from PIL import Image
import os
from processI import ProcessI
from rules import Rules

'''
it is used to generete train and test pickle files for train model and test model, we generate 2 kinds of train
pickle, train1.pkl is a larger train set adding extra images from kaggle, there are around 40 symboles and 1000 images
for each symbol. 
'''

rules = Rules().getrules()
data_train_root = os.getcwd() + "/data/train/"
data_test_root = os.getcwd() + "/data/test/"
data_mtrain_root = os.getcwd() + "/SelectedTrainingSet"

data_train = {}
data_test = {}
symbol_train = {}
symbol_test = {}
image_train = {}
image_test = {}

def gettrainsymbol():
    for f in os.listdir(data_train_root):
        if f.endswith(".png"):
            im1,im2 = ProcessI(data_train_root).processImage(f) #im1 is image data, im2 is image
            image_train[f] = im1
            symbol_train[f] = rules[f.split(".")[0].split("_")[3]]


    for subdir, dirs, files in os.walk(data_mtrain_root):
        for f in files:
            if f.endswith(".jpg"):
                # fileroot = os.path.join(subdir, f)
                im1,im2 = ProcessI(subdir+"/").processImage(f) #im1 is image data, im2 is image
                image_train[f] = im1
                e = subdir.split("/")
                # print e[len(e)-1]
                symbol_train[f] = rules[e[len(e)-1]]

    data_train["images"] = image_train
    data_train["labels"] = symbol_train
    pf = open("train1.pkl","wb")
    pickle.dump(data_train, pf)

def gettestsymbol():
    for f in os.listdir(data_test_root):
        if f.endswith(".png"):
            im1,im2 = ProcessI(data_test_root).processImage(f) #im1 is image data, im2 is image
            image_test[f] = im1
            symbol_test[f] = rules[f.split(".")[0].split("_")[3]]
    data_test["images"] = image_test
    data_test["labels"] = symbol_test
    pf = open("test.pkl","wb")
    pickle.dump(data_test, pf)

def main():
    gettrainsymbol()
    gettestsymbol()

if __name__ == "__main__":
    main()
