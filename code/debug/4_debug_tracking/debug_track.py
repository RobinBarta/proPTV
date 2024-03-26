'''

    Debug broken tracks.
    
'''


import os, shutil, joblib, h5py, sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

shutil.copy('../../main/functions/setup.py', 'functions/')
shutil.copy('../../main/functions/initialisation.py', 'functions/')
shutil.copy('../../main/functions/prediction.py', 'functions/')
shutil.copy('../../main/functions/tracking.py', 'functions/')
shutil.copy('../../main/functions/soloff.py', 'functions/')
from functions.setup import *
from functions.tracking import *
from functions.soloff import *

os.chdir('../../../data')


# %%

class ExtendParameter():    
    casename, runname, Zeros = '36000', 'run3', 5
    cams, t, dt = [0,1,2,3], 3, 1   
    
    Vmin, Vmax = [0,0,0], [1,1,1]
    
    d = 0.0075
    gaptracking = False                                    
    maxvel, activeMatches_extend, epsR = 0.015, 3, 3
    
# %%


def main(): 
    # load parameter
    params = ExtendParameter()
    params.load_path = params.casename+'/output/'+params.runname+'/tracks/'
    # load calibration
    ax = [np.loadtxt(params.casename+'/input/calibration/c{cam}/soloff_c{cam}{xy}.txt'.format(cam=cam,xy="x"),delimiter=',') for cam in params.cams]
    ay = [np.loadtxt(params.casename+'/input/calibration/c{cam}/soloff_c{cam}{xy}.txt'.format(cam=cam,xy="y"),delimiter=',') for cam in params.cams]
    # load img points
    Ps = np.loadtxt(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros)),skiprows=1)
    imgPs_true = np.asarray([np.vstack([F(Ps[:,:3],ax[c]),F(Ps[:,:3],ay[c])]).T for c in params.cams],dtype=object)
    imgPs = np.asarray([np.loadtxt(params.casename+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'.format(cam=c,time=str(params.t).zfill(params.Zeros)),skiprows=1) for c in params.cams],dtype=object)
    imgPs_gap = np.asarray([np.loadtxt(params.casename+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'.format(cam=c,time=str(params.t+1).zfill(params.Zeros)),skiprows=1) for c in params.cams],dtype=object)
    # load tracks
    data = h5py.File(params.load_path+"tracks_broken{time}.hdf5".format(time=str(params.t)), "r")
    allTracks = [data[key][:] for key in tqdm(list(data.keys()),desc=' loading tracks: ',leave=True,position=0,delay=0.5)]
    data.close()
    print(' ' + str(len(allTracks)) + ' broken tracks loaded at t = ' + str(params.t) + '\n' )
    
    if os.path.isfile(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros))):
        P_clouds_true = [np.loadtxt(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(t).zfill(params.Zeros)),skiprows=1) for t in range(params.t)]
        tracks_true = [ np.asarray([[P_clouds_true[j][i,1],P_clouds_true[j][i,2],P_clouds_true[j][i,3]] for j in range(params.t)]) for i in range(len(P_clouds_true[0])) ]
        # debug routine
        tracks, tracksj, Pt = allTracks.copy(), tracks_true.copy(), [P_clouds_true[i].copy() for i in range(len(P_clouds_true))]
        count, Is, Js = 0, [], []
        for i,track in enumerate(tqdm(allTracks,desc='Debug broken tracks: ',leave=True,position=0,delay=0.5)):
            IDS = []
            for t in range(params.t):
                ID = np.argwhere(np.linalg.norm( np.asarray(track)[t,1:4:] - Pt[t][:,1:4:] , axis=1) < params.d)[:,0]
                IDS.append(ID)
            if all(len(ID)>0 for ID in IDS):
                if np.std([ID[0] for ID in IDS])==0:
                    count+=1
                    Is.append(i)
                    Js.append(IDS[0][0])
        print(str(count) + ' / ' + str(len(allTracks)) + ' ( '+str(round(count/len(allTracks)*100,2))+' % ) correct tracks')
    
    # debug broken tracks
    allTracks = LoadBrokenTracks(params,'_broken'+str(params.t))
    allTracks_extend = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(ExtendTrack)(track,[],imgPs,imgPs_gap,ax,ay,params) for track in tqdm(allTracks,delay=0.5,position=0,leave=True,desc='Extend: '))
    allTracks_extended = [track for track in allTracks_extend if track[0][-1] == params.t]
    print(' extended ' + str(len(allTracks_extended)) + ' / ' + str(len(allTracks)) + ' broken tracks\n')
    
    # plot tracks
    fig, axis = plt.subplots(1,2,figsize=(16,8),subplot_kw=dict(projection='3d'))
    axis[0].set_title('broken tracks')        
    for track in tqdm(allTracks, desc='Plot tracks', position=0 , leave=True, delay=0.5):
        track = np.array(track[1])
        x, y, z = track[:,0], track[:,1], track[:,2]
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel/2,params.maxvel/2),linewidths=0.8,alpha=1)
        lc.set_array(Init_Velocity3D(track)[:,2]) 
        axis[0].add_collection3d(lc)
    axis[1].set_title('extended broken tracks')
    for track in tqdm(allTracks_extended, desc='Plot tracks', position=0 , leave=True, delay=0.5):
        track = np.array(track[1])
        x, y, z = track[:,0], track[:,1], track[:,2]
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel/2,params.maxvel/2),linewidths=0.8,alpha=1)
        lc.set_array(Init_Velocity3D(track)[:,2])
        axis[1].add_collection3d(lc)
    plt.show()
if __name__ == "__main__":
    main()