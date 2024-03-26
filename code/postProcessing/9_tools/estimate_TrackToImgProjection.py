'''

    Script to estimate track projection back into images.
    
'''


import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Track_parameter():    
    case_name, runname, suffix, Zeros = 'rbc_300mm_run2', 'proPTV__1_250', '', 7
    t_start, t_end, dt = 1 ,250, 1
    loadBroken = False
    
    showimg = True
    cam, t0, t1 = 0, 100, 107
    
# %%


def main(): 
    # load params
    params = Track_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    params.img_path = params.case_name+'/input/raw_images/c{cam}/c{cam}_{time}.tif'
    params.points_path = params.case_name+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    if params.loadBroken == True:
        for t in times:
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # load image
    ts = np.argwhere((params.t0<=times) & (times<=params.t1))[:,0] 
    img = []
    for t in times[ts]:
        imgtmp = cv2.imread(params.img_path.format(cam=params.cam,time=str(t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED)
        img.append(imgtmp)
    img = np.max(img,axis=0)
    imgpoints = [np.loadtxt(params.points_path.format(cam=params.cam,time=str(t).zfill(params.Zeros)),skiprows=1) for t in times[ts]]

    # plot 
    plt.rcParams.update({'font.family':'arial'})
    fig, axis = plt.subplots(figsize=(8,8))
    if params.showimg == True:
        plt.imshow(img,cmap='gray',vmax=np.mean(img[img!=0]))
    else:
        plt.imshow(img*0,cmap='gray')
        [plt.plot(imgpoints[i][:,0],imgpoints[i][:,1],'o',c='white') for i in range(len(imgpoints))]
    for track in tqdm(allTracks, desc='Project tracks', position=0 , leave=True, delay=0.5):
        ks = np.argwhere((params.t0<=track[:,0]) & (track[:,0]<=params.t1))[:,0] 
        if len(ks)>1:
            x = track[ks[0]:ks[-1]+1,10+int(2*params.cam)]
            y = track[ks[0]:ks[-1]+1,10+int(2*params.cam+1)]
            plt.plot(x,y,'.-',c='red')
    plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()  