'''

    This script estimates a list of image coordinates xyXYZ
    
'''


import os, cv2
import numpy as np

from getmarker_functions import *

os.chdir('../../../data/')


# %%

class Target_parameter:
    case_name, Zeros                = '5000', 5 #'rbc_300mm', 5
    t0, t1, cam, plane              = 1, 1, 0, 1
    alpha                           = 0.1
    
    threshold                       = 100 #950
    minArea , maxArea               = 30 , 200
    distance_line                   = 20
    
    N_marker                        = int(19*19)
    depth                           = ['x', [0. , 0.2, 0.4, 0.6, 0.8, 1. ]]            #['y',[26,  88, 150, 212, 274]] # [mm]
    startPoint                      = ['yz',0,0]    #['xz',15,15] # [mm]
    spacing                         = 1/18          #15 # [mm]

# %%
    
    
def main():
    params = Target_parameter()
    params.image_input = params.case_name+"/input/calibration_images/c{cam}/{plane}/calib_c{cam}_{plane}_{time}.tif"
    params.markerList_output = params.case_name+"/input/calibration_images/c{cam}/marker_c{cam}_{plane}.txt"
    
    # define ouput lists
    global mask_points, artifacts, artifacts_add, marker_lines, multiplier
    mask_points, artifacts, artifacts_add, marker_lines, multiplier = [], [], [], [], 1
    # load calibration target image
    times = np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int)
    try:
        img = np.min([cv2.imread(params.image_input.format(cam=params.cam,time=str(ti).zfill(params.Zeros),plane=params.plane),cv2.IMREAD_UNCHANGED) for ti in times], axis=0)
        # resize image to current screen
        img_resize, multiplier = Resize(img,params.alpha)
    except:
        img = np.min([cv2.cvtColor(cv2.imread(params.image_input.format(cam=params.cam,time=str(ti).zfill(params.Zeros),plane=params.plane),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2GRAY) for ti in times], axis=0)
        # resize image to current screen
        img_resize, multiplier = Resize(img,params.alpha)
    img_copy = img.copy()
    
    # create mask , by clicking: leftbot , lefttop, righttop, rightbot
    print('Select mask points: (right-click to finish)')
    mask_points = CollectMask(img_resize,mask_points,multiplier)
    img_masked = Masking(img,np.zeros(img.shape),np.asarray(mask_points))
    print('')
    
    # threshold image
    img_masked[img_masked<params.threshold] = 0
    img_thresh = cv2.convertScaleAbs(img_masked, alpha=params.alpha)
    
    # find markers on image
    print('Marker search: ')
    cx, cy = RadialSymmetricCenter(img_thresh,params) 
    print(' found ' + str(len(cx)) + ' / ' + str(params.N_marker) + '\n')
    
    # delete artifacts
    print('Select marker to delete: (ESC to finish)')
    cx_del, cy_del = DeleteArtifacts(cx,cy,img_thresh,artifacts,multiplier)
    print(' deleted ' + str(len(cx)-len(cx_del)) + ' marker\n')
    
    # search again
    print('Search new marker: (right-click to finish)')
    cx_add, cy_add = SearchArtifacts(cx_del,cy_del,img_thresh,artifacts_add,multiplier)
    cx, cy = np.append(cx_del,cx_add),np.append(cy_del,cy_add)
    print(' found ' + str(len(cx)) + ' total marker\n')
    
    # marker list sorting, from bot to top, by clicking left right
    print('Search corner marker (left(down,up) -> right(down,up)): (ESC to finish)')
    xyl, xyr = CollectMarkerPoints(img_thresh,cx,cy,marker_lines,multiplier)
    centers = np.vstack([cx,cy]).T
    marker_points = FindMarker(xyl,xyr,centers,img_copy,cx,cy,params)
    
    # output marker list
    np.savetxt(params.markerList_output.format(cam=params.cam,plane=params.plane),marker_points,header='x,y,X,Y,Z')
if __name__ == "__main__":
    main()
