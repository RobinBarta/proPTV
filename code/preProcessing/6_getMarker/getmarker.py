'''

    This script estimates a list of image coordinates xyXYZ
    
'''


import os, cv2
import numpy as np

from getmarker_functions import *

os.chdir('../../../data/')


# %%

class Target_parameter:
    case_name, Zeros                = 'Ilmenau', 5
    t0, t1, cam, plane              = 1, 1, 3, 3
    alpha                           = 0.1
    
    threshold                       = 50000
    minArea , maxArea               = 100 , 1000
    distance_line                   = 30
    
    N_marker                        = int(16*25)
    N_x, N_y                        = 25, 16
    depth                           = ['y',[-700,0,700]] # [mm]
    startPoint                      = ['xz',0,80] # [mm]
    spacing                         = 40 # [mm]

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
    print('Select mask points: (ESC to finish)')
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
    print('Search new marker: (ESC to finish)')
    cx_add, cy_add = SearchArtifacts(cx_del,cy_del,img_thresh,artifacts_add,multiplier)
    cx, cy = np.append(cx_del,cx_add),np.append(cy_del,cy_add)
    print(' found ' + str(len(cx)) + ' total marker\n')
    
    # marker list sorting, from bot to top, by clicking left right
    print('Search corner marker (left(down,up) -> right(down,up)): (ESC to finish)\n')
    xyl, xyr = CollectMarkerPoints(img_thresh,cx,cy,marker_lines,multiplier)
    centers = np.vstack([cx,cy]).T
    marker_points = FindMarker(xyl,xyr,centers,img_copy,cx,cy,params)
    
    # correct marker grid
    print('Correcting marker grid')
    marker_points_cor = Grid_correction(marker_points,img_copy,params)
    
    # output marker list
    np.savetxt(params.markerList_output.format(cam=params.cam,plane=params.plane),marker_points_cor,header='x,y,X,Y,Z')
if __name__ == "__main__":
    main()
