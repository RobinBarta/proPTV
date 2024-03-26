'''
    
    This script estimates the ppp value of a given data case.

'''


import cv2, os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('../../../data')


# %%

class InfoParameter():
    case_name, runname, Zeros = 'syn_8000_20', 'proPTV_8000_0_10', 5
    cam, t = 0 , 0
    
    # number of pixels per particle
    N = 3
    # intensity threshold
    thresh = 4000
    # section of the image x0:x0+dx in each direction
    x0, dx = 300, 100

# %%


def main(): 
    # load params
    params = InfoParameter()
    # load img
    img = cv2.imread(params.case_name+'/input/raw_images/c{cam}/c{cam}_{time}.tif'.format(cam=params.cam,time=str(params.t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) 
    
    # Threshold the image
    img[img<params.thresh] = 0
    # calculate ppp
    I = img[params.x0:params.x0+params.dx:,params.x0:params.x0+params.dx:]
    ppp = round( len(np.ravel(I[I>params.thresh])) / (params.dx*params.dx) / params.N , 4 )
    
    # plot output
    plt.figure(figsize=(8,8))
    plt.imshow(I,cmap='gray')
    plt.title('ppp = ' + str(ppp))
    plt.tight_layout(), plt.show()
    
    plt.savefig(params.case_name+'/analysis/ppp.tif')
if __name__ == "__main__":
    main()