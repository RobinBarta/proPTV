'''

    Debug image processing.
    
'''


import cv2, os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir('../../../data')

    
# %%

class DebugImgParameter():
    casename, Zeros = '18000', 5
    cams, t = [0], 5 
    d = 0.3
    
# %%


def main(): 
    # load parameter
    params = DebugImgParameter()
    
    # create output plot
    if os.path.isfile(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros))):
        fig, axis = plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    # for each camera perfrom debugging for given time step
    for ci,c in enumerate(params.cams):
        axis[0,ci].set_title('c' + str(c))
        # load image 
        img = cv2.imread(params.casename+'/input/raw_images/c{cam}/c{cam}_{time}.tif'.format(cam=c,time=str(params.t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) 
        axis[0,ci].imshow(img,cmap='gray')
        axis[1,ci].imshow(img,cmap='gray')
        # load image particle list
        imgPs = np.loadtxt(params.casename+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'.format(cam=c,time=str(params.t).zfill(params.Zeros)),skiprows=1)
        axis[0,ci].plot(imgPs[:,0],imgPs[:,1],'o',c='red')
        print(len(imgPs)-len(np.unique(imgPs,axis=0)))
        # load ground truth if it is a syn case
        if os.path.isfile(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros))):
            Ps = np.loadtxt(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros)),skiprows=1)[:,9+int(2*c):9+int(2*c+1)+1]
            axis[0,ci].plot(Ps[:,0],Ps[:,1],'.',c='green')
            # run debug and calculate all matched particles
            x, y, dels = Ps[:,0].copy(), Ps[:,1].copy(), []
            for i, p in enumerate(tqdm(imgPs,desc='Debug Image Processing at c = ' + str(c),position=0,leave=True,delay=0.5)):
                dP = np.sqrt((p[0]-x)**2+(p[1]-y)**2)
                IDs = np.argwhere( dP < params.d )[:,0]
                if len(IDs)>0:
                    dels.append(i)
                    ID = IDs[np.argmin(dP[IDs])]
                    x, y = np.delete(x,ID,axis=0), np.delete(y,ID,axis=0)
            imgPs_del = np.delete(imgPs.copy(),dels,axis=0)
            axis[1,ci].plot(imgPs_del[:,0],imgPs_del[:,1],'o',c='red')
            axis[1,ci].plot(x,y,'.',c='green')
            print(' hit points: ' + str(len(Ps)-len(x)) + ' / ' + str(len(Ps)) + ' ( ' + str(np.round((len(Ps)-len(x))/len(Ps)*100,2)) + ' % )')
            print(' wrong points: ' + str(len(imgPs_del)) + ' / ' + str(len(imgPs)) + ' ( ' + str(np.round(len(imgPs_del)/(len(Ps)-len(x))*100,2)) + ' % )\n')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()