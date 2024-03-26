'''
    Script to rename files.
    Enter parameters and run script.
'''

import os, cv2
import numpy as np

from tqdm import tqdm
from pco_tools import pco_reader as pco

os.chdir('../../../data')

# %%

class Rename_parameter:
    # path to folder where files are renamed
    case_name = "rbc_500mm"
    # file characteristics, like name, time strings etc.
    t_start, t_end = 2000, 2100
    cams = [0,1,2,3]
    oldName, newName = "c{cam}_{time}.b16", "c{cam}_{time}.tif"
    Zeros_old, Zeros_new = 4, 5

# %%

def main():
    # load params
    params = Rename_parameter()
    params.input_path = params.case_name+"/input/raw_images/c{cam}/"+params.oldName
    params.output_path = params.case_name+"/input/raw_images/c{cam}/"+params.newName
    
    # for each file make a tif
    for t in tqdm(np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int),desc='Make TIF',position=0,leave=True,delay=0.5):
        for i in range(len(params.cams)):
            if params.oldName[-3::] == 'b16':
                img = pco.load(params.input_path.format(cam=params.cams[i],time=str(t).zfill(params.Zeros_old)))
                cv2.imwrite(params.output_path.format(cam=params.cams[i],time=str(t).zfill(params.Zeros_new)),img)
if __name__ == "__main__":
    main()