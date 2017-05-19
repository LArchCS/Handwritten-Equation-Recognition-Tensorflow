import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import os
import pickle
import glob
import pprint
import operator

sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']

slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'sin', 'pm', 'frac', 'cos', 'delta', 'bar', 'div','^','_']

variable = ['1', '0', '3', '2', '4', '6', 'pi', 'A', 'a', 'c', 'b', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', '(', ')']
brules = {}

def imageprepare(image):
    im = image.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0)) #creates black canvas of 28x28 pixels

    if width > height: #check which dimension is bigger
        nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
        img = im.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (0, wtop)) #paste resized image
    else:
        nwidth = int(round((28.0/height*width),0)) #resize width according to ratio height
        img = im.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 0)) #paste resized image on
    tv = list(newImage.getdata()) #get pixel values
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ 1-(255-x)*1.0/255.0 for x in tv]
    return tva, newImage

def update(im_name, symbol_list):
    im = Image.open(im_name)
    list_len = len(symbol_list)
    for i in range(list_len):
        if i >= len(symbol_list): break
        
        symbol = symbol_list[i]
        predict_result = symbol[1]
        
        # deal with equal mark
        if predict_result == "-":
            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1] 
                if s1[1] == "-" and abs(s1[2] - symbol[2]) < 30 and abs(s1[4] - symbol[4]) < 30:
                    updateEqual(symbol, s1, symbol_list, im, i)
                    continue
        
        # deal with bar
        if predict_result == "-":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if isVSame(symbol, s1) and (not isVSame(symbol, s2)):
                    updateBar(symbol, symbol_list, im, i)
                    continue
        
        # deal with division mark
        if predict_result == "-":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2] 
                if s1[3] < symbol[3] and s2[3] > symbol[3] and (s2[2] - s1[2]) < 30:
                    if s1[1] == "dot" and s2[1] == "dot":
                        updateDivision(symbol, s1, s2, symbol_list, im, i)
                        continue
        
        # deal with fraction
        if predict_result == "-":
            j = i
            upPart = 0
            underPart = 0
            while j < len(symbol_list):
                tmp = symbol_list[j]
                if tmp[2] > symbol[2] and tmp[4] < symbol[4] and tmp[5] > symbol[3]: upPart += 1
                if tmp[2] > symbol[2] and tmp[4] < symbol[4] and tmp[3] < symbol[5]: underPart += 1
                j += 1
            if upPart > 0 and underPart > 0:
                updateFrac(symbol, symbol_list, im, i)
                continue
                        
        # deal with dots
        if predict_result == "dot":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if symbol_list[i+1][1] == "dot" and symbol_list[i+2][1] == "dot":
                    updateDots(symbol, s1, s2, symbol_list, im, i)
                    continue
        
        # deal with i
        if predict_result == "dot":
            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1]
                if s1[1] == "1" and abs(s1[2] - symbol[2]) < 30:
                    updateI(symbol, s1, symbol_list, im, i)
                    continue
            
            if i > 1:
                s1 = symbol_list[i-1]
                if s1[1] == "1" and abs(s1[2] - symbol[2]) < 30:
                    updateI(symbol, s1, symbol_list, im, i)
                    continue
                
        # deal with +-
        if i < (len(symbol_list) - 1):
            if (symbol[1] == "+" and symbol_list[i+1][1] == "-") or (symbol[1] == "-" and symbol_list[i+1][1] == "+"):
                x,y,xw,yh = symbol[2:]
                x1, y1, xw1, yh1 = symbol_list[i+1][2:]
                cenX = x + (xw - x) / 2
                cenX1 = x1 + (xw1 - x1) / 2
                s1 = symbol_list[i+1]
                if abs(cenX - cenX1) < 15:
                    updatePM(symbol, s1, symbol_list, im, i)
                    continue            
        
    return symbol_list

def toLatex(symbol_list):
    s = []
    i = 0
    while (i < len(symbol_list)):
        symbol = symbol_list[i]
        value = symbol[1]
        
        if value == 'frac':
            upper = []
            under = []
            i = i + 1
            while (i < len(symbol_list) and (isUpperFrac(symbol, symbol_list[i]) or isUnderFrac(symbol, symbol_list[i]))):
                if isUpperFrac(symbol, symbol_list[i]): upper.append(symbol_list[i])
                if isUnderFrac(symbol, symbol_list[i]): under.append(symbol_list[i])
                i = i + 1
            if len(upper) > 0 and upper[len(upper) - 1][1] not in variable:
                upper.pop()
                i = i - 1
            if len(under) > 0 and under[len(under) - 1][1] not in variable:
                under.pop()
                i = i - 1
            upper_string = '{' + toLatex(upper) + '}'
            under_string = '{' + toLatex(under) + '}'
            s.append('\\frac'+upper_string+under_string)
            continue
        elif value == 'sqrt':
            outer = []
            inner = []
            i = i + 1
            while (i < len(symbol_list) and isInner(symbol, symbol_list[i])):
                inner.append(symbol_list[i])
                i = i + 1
            if len(inner) > 0 and inner[len(inner) - 1][1] not in variable:
                inner.pop()
                i = i - 1
            inner_string = '{' + toLatex(inner) + '}'
            s.append('\\sqrt'+inner_string)
            continue
        elif value in slash_sy: 
            s.append('\\' + value)
            base = i
        elif i > 0 and (s[len(s) - 1] in slash_sy): 
            # need to consider about range within squrt and frac
            s.append('{'+value+'}')
        elif i < len(symbol_list) - 1 and isUpperSymbol(symbol, symbol_list[i+1]) and (symbol[1] in variable) and (symbol_list[i+1][1] in variable): 
            s.append(value)
            s.append('^{')
            i = i+1
            while (i < len(symbol_list) and isUpperSymbol(symbol, symbol_list[i])):
                s.append(symbol_list[i][1])
                i = i + 1
            s.append('}')
            continue
        elif i < len(symbol_list) - 1 and isLowerSymbol(symbol, symbol_list[i+1]) and (symbol[1] in variable) and (symbol_list[i+1][1] in variable): 
            s.append(value)
            s.append('_{')
            i = i+1
            while (i < len(symbol_list) and isLowerSymbol(symbol, symbol_list[i])):
                s.append(symbol_list[i][1])
                i = i + 1
            s.append('}')
            continue
        else: 
            s.append(value)
            base = i
        i = i + 1
    return "".join(s)
                    
def isVSame(cur, next):
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    next_center_x = next[2] + (next[4] - next[2])/2
    if abs(cur_center_x - next_center_x) < 30: return True
    else: return False

def isInner(cur, next):
    if next[3] < cur[5] and next[2] > cur[2] and next[4] - cur[4] < 10: return True
    else: return False
    
def isUpperFrac(cur, next):
    if next[5] < cur[3] and next[2] - cur[2] > -10 and next[4] - cur[4] < 10: return True
    else: return False

def isUnderFrac(cur, next):
    if next[3] > cur[5] and next[2] - cur[2] > -10 and next[4] - cur[4] < 10: return True
    else: return False
    
def isUpperSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3])/2
    next_center = next[3] + (next[5] - next[3])/2
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    if next_center < cur_center - (next[5] - next[3])/2 and next[2] > cur_center_x: return True 
    else: return False

def isLowerSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3])/2
    next_center = next[3] + (next[5] - next[3])/2
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    if next_center > cur_center + (next[5] - next[3])/2 and next[2] > cur_center_x: return True
    else: return False

def area(symbol):
    return (symbol[4] - symbol[2]) * (symbol[5] - symbol[3])

def updateEqual(symbol,s1,symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "=", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)
    
def updateDivision(symbol,s1,s2,symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "div", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)
    
def updateDots(symbol,s1,s2,symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "dots", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)
    
def updateI(symbol,s1,symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "i", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)
    
def updatePM(symbol,s1,symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "pm", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)
    
def updateBar(symbol,symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "bar", x, y, xw, yh)
    symbol_list[i] = new_symbol
    
def updateFrac(symbol,symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "frac", x, y, xw, yh)
    symbol_list[i] = new_symbol