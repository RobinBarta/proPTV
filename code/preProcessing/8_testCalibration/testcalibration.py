'''
    This script runs a calibration test.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

from testcalibration_functions import *

# %%

class TestCalibrationParameter:
    cams = [0,1,2,3]
    x0 , x1 , Nx = 0 , 300 , 10
    y0 , y1 , Ny = 0 , 300 , 10
    z0 , z1 , Nz = 0 , 300 , 10
        
    case = "rbc_300mm_run2"
    rawimage_input = "../../../data/"+case+"/input/raw_images/c{cam}/c{cam}_0000001.tif"
    markerList_input = "../../../data/"+case+"/input/calibration_images/marker_c{cam}.txt"
    calibration_path = "../../../data/"+case+"/input/calibration/c{cam}/soloff_c{cam}{xy}.txt"

# %%

def main():
    params = TestCalibrationParameter()
    ax = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="x"),delimiter=',') for c in params.cams])
    ay = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="y"),delimiter=',') for c in params.cams])

    i = 0
    n = [0,1,0,1]
    m = [0,0,1,1]
    
    # 2D plot
    fig, axis = plt.subplots(2,2,sharex=True,sharey=True)
    for c in params.cams:
        X , Y , Z = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.y0,params.y1,params.Ny),np.linspace(params.z0,params.z1,params.Nz))
        XYZ = np.vstack([np.ravel(X),np.ravel(Y),np.ravel(Z)]).T
        #XYZ = np.loadtxt(params.markerList_input.format(cam=c),skiprows=1)[:,2::]        
        x , y = Soloff(XYZ, ax[c]) , Soloff(XYZ, ay[c])
        img = cv2.imread(params.rawimage_input.format(cam=c),cv2.IMREAD_UNCHANGED)
        axis[n[i]][m[i]].imshow(img,cmap='gray')
        axis[n[i]][m[i]].plot(x,y,'.',c='red')
        i+=1
    plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()