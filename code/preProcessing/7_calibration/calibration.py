'''
    This script does the calibration based on an marker list per camera.
'''


import os
import numpy as np

from scipy.optimize import least_squares

from calibration_functions import *

os.chdir("../../../data/")


# %%

class Calibration_parameter:
    case_name = 'rbc_300mm'
    cams = [0,1,2,3]

# %%
  
  
def main():
    params = Calibration_parameter()
    params.markerList_input = params.case_name+"/input/calibration_images/markers_c{cam}.txt"
    params.calibration_output = params.case_name+"/input/calibration/c{cam}/soloff_c{cam}{xy}.txt"
    
    # load marker lists for each camera
    for cam in params.cams:
        data = np.loadtxt(params.markerList_input.format(cam=cam),skiprows=1)
        # least square estimation of the Soloff parameter
        sx , sy = Calibration(data, [np.zeros(19),np.zeros(19)])
        np.savetxt(params.calibration_output.format(cam=cam,xy="x"),sx)
        np.savetxt(params.calibration_output.format(cam=cam,xy="y"),sy)
if __name__ == "__main__":
    main()