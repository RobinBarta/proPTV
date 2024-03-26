'''

    This script combines the marker list per camera
    
'''


import os
import numpy as np

os.chdir('../../../data/')


# %%

class Target_parameter:
    case_name = 'rbc_300mm'
    cams, planes = [0,1,2,3], [1,2,3,4,5]

# %%
    
    
def main():
    params = Target_parameter()
    params.markerList_input = params.case_name+"/input/calibration_images/c{cam}/marker_c{cam}_{plane}.txt"
    params.markerList_output = params.case_name+"/input/calibration_images/markers_c{cam}.txt"
    
    for c in params.cams:
        markerlist = np.empty([0,5])
        for i in params.planes:
            markers = np.loadtxt(params.markerList_input.format(cam=c,plane=i),skiprows=1)
            markerlist = np.append(markerlist,markers,axis=0)
        np.savetxt(params.markerList_output.format(cam=c),markerlist,header='x,y,X,Y,Z')
if __name__ == "__main__":
    main()