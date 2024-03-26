import cv2, sys
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.draw import polygon
from pco_tools import pco_reader as pco

def Resize(img,params):
    root = tk.Tk()
    h, w = img.shape
    screen_h, screen_w = root.winfo_screenheight(), root.winfo_screenwidth()
    window_h, window_w = screen_h*np.sqrt(0.8), screen_w*np.sqrt(0.8)
    img_resize = cv2.convertScaleAbs(img, alpha=params.alpha)
    multiplier = 1
    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
            multiplier = window_h / h
        else:
            multiplier = window_w / w
        img_resize = cv2.convertScaleAbs(cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier), alpha=params.alpha)
    return img_resize, multiplier

def click_event1(event, x, y, flags, param):
    global mask_points, multiplier
    mask_points, multiplier = param[0], param[1]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        mask_points.append([int(np.round(y/multiplier)),int(np.round(x/multiplier))])
        print('Corner: x = '+str(x) , 'y = '+str(y))
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()
        
def CollectMask(img_resize,mask_points,multiplier):
    cv2.imshow('Get Mask', img_resize)
    cv2.setMouseCallback('Get Mask', click_event1, [mask_points, multiplier])
    cv2.waitKey(0)
    return np.asarray(mask_points)