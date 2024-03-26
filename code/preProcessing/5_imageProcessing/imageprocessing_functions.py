'''

    This script contains functions for image processing

'''


import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from scipy.signal import convolve2d

def Get_IDs(center_x, center_y, step_width, size):
    surrounding_ids = []
    for y in range(center_y - step_width, center_y + step_width + 1):
        for x in range(center_x - step_width, center_x + step_width + 1):
            distance = ((y - center_y) ** 2 + (x - center_x) ** 2) ** 0.5
            if abs(distance - step_width) < size:
                surrounding_ids.append([y,x])
    return np.asarray(surrounding_ids)

def ImageProcessing(cam, t, i, times, params):
    # load image
    img_origin = cv2.imread(params.image_input.format(cam=cam,time=str(t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED)
    # create processing img
    img = img_origin.copy()
    
    # averaging           
    a = i-(params.window//2) if i>=(params.window//2) else 0 
    a = len(times)-params.window if i>=(len(times)-(params.window//2)) else a 
    b = i+(params.window//2) if i<(len(times)-(params.window//2)) else len(times)-1
    b = params.window-1 if i<(params.window//2) else b
    min_img = params.weight_min*np.min([cv2.imread(params.image_input.format(cam=cam,time=str(ti).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) for ti in times[a:b+1]], axis=0)
    img = img-min_img
    
    # thresholding
    img[min_img>params.threshold_minimg] = 0
    img[img<params.threshold] = 0
    
    # masking
    mask = cv2.imread(params.mask_path.format(cam=cam),cv2.IMREAD_UNCHANGED)
    img[mask==0] = 0
    
    # delete single artifacts
    if params.delete_artifacts == True:
        img[convolve2d(img, [[1,1,1],[1,0,1],[1,1,1]], mode='same')==0] = 0
        
    # blur
    if params.blur == True:
        img = cv2.GaussianBlur(img, [3,3], 1)
    img = np.rint(img) 

    # create particle list for peak detection
    finalList = []
    img_peak = img.copy().astype('float')
    
    # calculate intensity distribution
    Imean = [np.mean(img[img>0])]
    for i in range(params.particleSize+1):
        Imean.append( Imean[0] * np.exp( - ((float(i)/(np.sqrt(2)*params.std))**2 + (0.0/(np.sqrt(2)*params.std))**2) ) )
    
    # peak search
    for n in range(params.runs):
        # peak detection
        particleList = peak_local_max(img_peak, min_distance=int(params.particleSize),num_peaks=params.maxParticle-len(finalList))
        CX , CY = [] , []
        for x,y in zip(particleList[:,1],particleList[:,0]):
            binsX, binsY = np.array([x,x+1,x-1],dtype=int), np.array([y,y+1,y-1],dtype=int)
            valueX , valueY = img_origin[y,binsX] , img_origin[binsY,x]
            meanX, meanY = np.sum(binsX*valueX) / np.sum(valueX), np.sum(binsY*valueY) / np.sum(valueY)
            CX.append(meanX) , CY.append(meanY)
        particleList = np.vstack([CX,CY]).T
        # subtract found images
        for p in np.rint(particleList).astype('int'):
            I_sub = np.zeros_like(img_peak)
            for i in range(params.particleSize+1):
                IDs = Get_IDs(p[0],p[1], i, 1)
                I_sub[IDs[:,0],IDs[:,1]] = Imean[i]
            img_peak -= I_sub
        img_peak[img_peak<params.threshold] = 0
        if len(particleList)>0:
           finalList += list(particleList)
    finalList = np.asarray(finalList)
    
    # split final list and correct image position
    d = params.std/2
    finalList_unique, IDs_unique = np.unique(finalList,axis=0,return_index=True)
    finalList_rest = np.delete(finalList,IDs_unique ,axis=0)
    for i, p in enumerate(finalList_rest):
        x, y = int(np.rint(p[0])), int(np.rint(p[1]))
        IDs = Get_IDs(x, y, 1, 0.4)
        ID = IDs[np.argmax(img_origin[IDs[:,0],IDs[:,1]])]
        dx, dy = d*(ID[1]-p[0]), d*(ID[0]-p[1])
        finalList_rest[i] = np.array([p[0]+dx, p[1]+dy])
    finalList = np.append(finalList_unique,finalList_rest,axis=0)
    
    finalList_unique, IDs_unique = np.unique(finalList,axis=0,return_index=True)
    finalList_rest = np.delete(finalList,IDs_unique ,axis=0)
        
    img_proc = np.zeros_like(img).astype('uint8')
    img_proc[np.asarray(np.round(CY),dtype=int),np.asarray(np.round(CX),dtype=int)] = 250
    img_proc = cv2.GaussianBlur( img_proc , [3,3] , 1 )*3
    
    # save proc image and particle list
    if params.debug == False:
        cv2.imwrite(params.image_output.format(cam=cam,time=str(t).zfill(params.Zeros)) , img_proc)
        np.savetxt( params.particleList_output.format(cam=cam,time=str(t).zfill(params.Zeros)) , finalList , header="center_x , center_y")
    return finalList, img_origin, img, img_proc, min_img