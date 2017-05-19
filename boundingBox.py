import PIL.ImageOps
import numpy as np
import cv2
import glob
import os
import scipy.misc

from scipy.misc import imsave
from PIL import Image, ImageDraw

equal_path = "./data/annotated_test_Equal/"
equal_result_path = "./data/annotated_test_Equal_result/"
equal_boxed_path = "./data/annotated_test_Equal_boxes/"

# clipt image into box using OpenCV

# The next step is further optimizing using the size provided by TA
# TA input size: 1696 * 117
# isFraction  - not updated

# detect if input boundingBox contains a dot
def isDot(boundingBox):
    (x, y), (xw, yh) = boundingBox
    area = (yh - y) * (xw - x)
    return area < 200 and 0.5 < (xw - x)/(yh - y) < 2 and abs(xw - x) < 20 and abs(yh - y) < 20  # 100 is migical number

# detect if input boundingBox contains a vertical bar
def isVerticalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (yh - y) / (xw - x) > 2

# detect if a given boundingBox contains a horizontal bar
def isHorizontalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) / (yh - y) > 2

# detect if input boundingBox contains a square (regular letters, numbers, operators)
def isSquare(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) > 8 and (yh - y) > 8 and 0.5 < (xw - x)/(yh - y) < 2

# detect if input three boundingBoxes are a division mark
def isDivisionMark(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2
    return (isHorizontalBar(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2)
            and x < x1 < x2 < xw and max(y1, y2) > y and min(y1, y2) < y
            and max(y1, y2) - min(y1, y2) < 1.2 * abs(xw - x))

# detect if input two boundingBoxes are a lowercase i
def isLetterI(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    return (((isDot(boundingBox) and isVerticalBar(boundingBox1)) or (isDot(boundingBox1) and isVerticalBar(boundingBox)))
            and abs(x1 - x) < 10)  # 10 is a magical number

# detect if input two boundingBoxes are an equation mark
def isEquationMark(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    return isHorizontalBar(boundingBox) and isHorizontalBar(boundingBox1) and abs(x1 - x) < 20 and abs(xw1 - xw) < 20 # 20 is a migical number

# detect if input three boundingBoxes are a ellipsis (three dots)
def isDots(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenY = y + (yh - y) / 2
    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2
    return (isDot(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2) and max(cenY, cenY1, cenY2) - min(cenY, cenY1, cenY2) < 50)  # 30 is a migical number

# detect if input two boundingBoxes are a plus-minus
def isPM(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    case1 = isHorizontalBar(boundingBox) and isSquare(boundingBox1) and x < cenX1 < xw and -15 < y - yh1 < 35 and xw - cenX1 < 50
    case2 = isSquare(boundingBox) and isHorizontalBar(boundingBox1) and x1 < cenX < xw1 and -15 < y1 - yh < 35 and xw1 - cenX < 50
    return case1 or case2  # magical number

# detect if input three boundingBoxes are a fraction
def isFraction(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    cenX2 = x2 + (xw2 - x2) / 2
    case1 = not isDot(boundingBox) and not isDot(boundingBox1) and isHorizontalBar(boundingBox2) and (y < y2 < yh1 or y1 < y2 < yh)
    case2 = not isDot(boundingBox2) and not isDot(boundingBox) and isHorizontalBar(boundingBox1) and (y2 < y1 < yh or y < y1 < yh2)
    case3 = not isDot(boundingBox1) and not isDot(boundingBox2) and isHorizontalBar(boundingBox) and (y1 < y < yh2 or y2 < y < yh1)
    return (case1 or case2 or case3) and  max(cenX, cenX1, cenX2) - min(cenX, cenX1, cenX2) < 50  # 30 is a migical number

# return initial bounding boxes of input image
def initialBoxes(im):
    '''input: image; return: None'''

    im[im >= 127] = 255
    im[im < 127] = 0
    
    '''
    # set the morphology kernel size, the number in tuple is the bold pixel size
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    '''

    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    im2, contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL for only bounding outer box
    # bounding rectangle outside the individual element in image
    res = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # exclude the whole size image and noisy point
        if x is 0: continue
        if w*h < 25: continue
            
        res.append([(x,y), (x+w, y+h)])
    return res

# take in raw bounding boxes and detect components should be connected
def connect(im, res):
    '''input: image, raw rectangles; return: joint rectangles indicating detected symbols'''
    finalRes = []
    res.sort()
    i = 0
    while (i < len(res) - 1):
        (x, y), (xw, yh) = res[i]
        (x1, y1), (xw1, yh1) = res[i+1]
#         print([(x, y), (xw, yh)], [(x1, y1), (xw1, yh1)])

        equation = isEquationMark(res[i],  res[i + 1])
        letterI = isLetterI(res[i], res[i+1])
        pm = isPM(res[i], res[i+1])
        divisionMark = False
        dots = False
        fraction = False
        if i < len(res) - 2:
            (x2, y2), (xw2, yh2) = res[i+2]
#             print([(x2, y2), (xw2, yh2)])
            divisionMark = isDivisionMark(res[i], res[i+1], res[i+2])
            dots = isDots(res[i], res[i+1], res[i+2])
            fraction = isFraction(res[i], res[i+1], res[i+2])

        # PM os really hard to determine, mixed with fraction
        if (equation or letterI or pm) and not fraction:
            finalRes.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
            i += 2
        elif (divisionMark or dots) and not fraction:
            finalRes.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
            i += 3
        else:
            finalRes.append(res[i])
            i += 1

    while i < len(res):
#         print([res[i][0], res[i][1]])
        finalRes.append(res[i])
        i += 1

    return finalRes

# slices im into smaller images based on boxes
def saveImages(im, boxes, im_name):
    '''input: image, boxes; return: None'''
    # make a tmpelate image for next crop
    image = Image.fromarray(im)
    num = 1
    boxes = sorted(boxes, key=lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
    for box in boxes:
        (x, y), (xw, yh) = box
        x -= 1
        y -= 1
        xw += 1
        yh += 1
        w = xw - x - 2
        h = yh - y - 2
        symbolImage = image.crop((x, y, xw, yh))
        # extract dot image to final result
        if w < 25 and h < 25 and float(w)/h < 1.5 and float(h)/w < 1.5 :
            symbolImage.save(equal_boxed_path + im_name + "_" + "dot" + "_" + str(y) + "_" + str(yh) + "_" + str(x) + "_" + str(xw) + ".png")
        else :
            # save rectangled element
            symbolImage.save(equal_boxed_path + im_name + "_" + str(num) + "_" + str(y) + "_" + str(yh) + "_" + str(x) + "_" + str(xw) + ".png")
        # fill the found part with black to reduce effect to other crop
        draw = ImageDraw.Draw(image)
        draw.rectangle((x, y, xw, yh), fill = 'black')
        # draw rectangle around element in image for confirming result
        cv2.rectangle(im, (x,y), (xw, yh), (0,255,0), 2)
        num = num + 1
    new_image = Image.fromarray(im)   
    new_image.save(equal_result_path + im_name + ".png")
        
# slices im into smaller images based on boxes
def createSymbol(path):
    '''input: image, boxes; return: None'''
    # make a tmpelate image for next crop
    im = cv2.imread(path)
    image = Image.fromarray(im)
    rawRes = initialBoxes(im)  # raw bounding boxes
    boxes = connect(im, rawRes)
    boxes = sorted(boxes, key=lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
    
    symbol_list= []
    for box in boxes:
        (x, y), (xw, yh) = box
        x -= 1
        y -= 1
        xw += 1
        yh += 1
        w = xw - x - 2
        h = yh - y - 2
        # save rectangled element
        symbolImage = image.crop((x, y, xw, yh))
        if w < 25 and h < 25 and float(w)/h < 1.5 and float(h)/w < 1.5 :
            symbol_info = (symbolImage, "dot", x, y, xw, yh);
        else :
            symbol_info = (symbolImage, "unknown", x, y, xw, yh);
        symbol_list.append(symbol_info)
        # fill the found part with black to reduce effect to other crop
        draw = ImageDraw.Draw(image)
        draw.rectangle((x, y, xw, yh), fill = 'black')
    return symbol_list

# run the code
def main():
    image_list = glob.glob(equal_path + "*.png")
    for im_name in image_list:
        head, tail = os.path.split(im_name)
        im = cv2.imread(im_name)  # specify the image to process
        rawRes =  initialBoxes(im)  # raw bounding boxes
        finalRes = connect(im, rawRes)  # connect i, division mark, equation mark, ellipsis
        image_name = os.path.splitext(tail)[0]
        saveImages(im, finalRes, image_name)


if __name__ == "__main__":
    main()