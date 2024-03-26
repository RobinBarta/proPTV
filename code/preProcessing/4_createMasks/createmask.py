'''
    This script creates 8bit masks for each camera by selecting image points.
    Enter parameter and run the script. Points selected via left mouse click in order: 
        left bottom , left top , right top , right bottom.
    Finish point selection with right mouse click
    Inside the mask spanned, the image takes the value 255 and outside the value 0.
'''

import cv2, sys
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.draw import polygon
from pco_tools import pco_reader as pco

from createmask_functions import *

# %%

class Mask_parameter:
    # select camera and time step
    cam, t = 3, 2000
    # adjust brightness of resized image
    alpha = 1#0.08
    
    # path params
    zFill = 4
    case = 'RBC500'
    raw_image = "../../../data/"+case+"/input/raw_images/c{cam}/c{cam}_"+str(t).zfill(zFill)+".tif"
    mask_output = "../../../data/"+case+"/input/masks/c{cam}/c{cam}_mask.tif"

# %%

def main():
    params = Mask_parameter()
    
    # load image 
    if params.raw_image[-3::] == 'b16':
        img = pco.load(params.raw_image.format(cam=params.cam))
    elif params.raw_image[-3::] == 'tif':
        img = cv2.imread(params.raw_image.format(cam=params.cam),cv2.IMREAD_UNCHANGED)
    # create raw mask
    raw = np.asarray(np.zeros_like(img),'uint8')
      
    # resize image to current screen
    img_resize, multiplier = Resize(img,params)
    # create mask by clicking: leftbot , lefttop, righttop, rightbot
    mask_points = []
    points = np.asarray(CollectMask(img_resize,mask_points,multiplier))
    # fill the mask
    rr, cc = polygon(points[:,0], points[:,1], raw.shape)
    raw[rr,cc] = 255
    
    # save mask
    cv2.imwrite(params.mask_output.format(cam=params.cam),raw)
if __name__ == "__main__":
    main()