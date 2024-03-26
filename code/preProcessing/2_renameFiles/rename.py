'''

    Script to rename files.
    
'''

import os
import numpy as np

# %%

class Rename_parameter:
    # path to folder where files are renamed
    path = "../../../data/rbc_300mm_run2/input/raw_images/c{cam}/"
    # file characteristics, like name, time strings etc.
    t_start, t_end = 20000, 20010
    cam_input, cam_output = [1,2,3,4],  [0,1,2,3]
    oldName, newName = "c{cam}_{time}.tif", "c{cam}_{time}.tif"
    Zeros_old, Zeros_new = 7, 7

# %%

def main():
    # load parameter
    params = Rename_parameter()
    # rename files
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    for t in times:
        for i in range(len(params.cam_input)):
            os.rename(params.path.format(cam=params.cam_output[i]) + params.oldName.format(cam=params.cam_input[i],time=str(t).zfill(params.Zeros_old)), 
                      params.path.format(cam=params.cam_output[i]) + params.newName.format(cam=params.cam_output[i],time=str(t).zfill(params.Zeros_new)))
if __name__ == "__main__":
    main()