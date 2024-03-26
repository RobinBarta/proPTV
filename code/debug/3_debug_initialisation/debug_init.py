'''

    Debug initialisation.
    
'''


import os, shutil, h5py
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

#shutil.copy('../../main/functions/initialisation.py', 'functions/')
#shutil.copy('../../main/functions/setup.py','functions/')
from functions.initialisation import *
from functions.setup import *

os.chdir('../../../data')


# %%

class LinkingParameter():    
    casename, runname, Zeros = "18000", "run1", 5
    cams = [0,1,2,3] 
    t0, t1, t_init, dt = 0, 2, 3, 1     
    N_init, NN, maxvel, angle = 3, [3,3,3], 0.015, 80    
    
    d = 0.0075
    
# %%


def main(): 
    # load parameter
    params = LinkingParameter()
    params.track_path = "../code/debug/3_debug_initialisation/output/"  
    # load triangulation
    P_clouds = [np.loadtxt(params.casename+"/output/"+params.runname+"/triangulation/Points_{time}.txt".format(time=str(t).zfill(params.Zeros)),skiprows=1) for t in np.linspace(params.t0,params.t1,params.t_init,dtype=int)[::params.dt]]
    print('triangulated particles per frame: ' , str([len(P) for P in P_clouds]))
        
    # do initialisation
    times_init = np.linspace(params.t0,params.t1,params.t_init,dtype=int)[::params.dt]
    print('\nInitialisation:')
    allTracks, P_clouds = Initialisation(P_clouds,times_init[-1],params)
    SaveTracks(allTracks, params, 'initial' ,params.t0 ,'w')
    data = h5py.File(params.track_path+"tracksinitial.hdf5", "r")
    allTracks = [data[key][:] for key in tqdm(list(data.keys()),desc=' loading tracks: ',leave=True,position=0,delay=0.5)]
    data.close()
    
    #data = h5py.File(params.casename+"/output/"+params.runname+"/tracks/tracks_initial.hdf5", "r")
    #allTracks = [data[key][:] for key in tqdm(list(data.keys()),desc=' loading tracks: ',leave=True,position=0,delay=0.5)]
    #data.close()
    print(' initialised ' +  str(len(allTracks)) + ' tracks\n')
    # debug initialisation from ground truth
    if os.path.isfile(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(times_init[0]).zfill(params.Zeros))):
        P_clouds_true = [np.loadtxt(params.casename+'/analysis/origin/origin_{time}.txt'.format(time=str(t).zfill(params.Zeros)),skiprows=1) for t in times_init]
        tracks_true = [ np.asarray([[P_clouds_true[j][i,1],P_clouds_true[j][i,2],P_clouds_true[j][i,3]] for j in range(params.t_init)]) for i in range(len(P_clouds_true[0])) ]
        # debug routine
        tracks, tracksj, Pt = allTracks.copy(), tracks_true.copy(), [P_clouds_true[i].copy() for i in range(len(P_clouds_true))]
        count, Is, Js = 0, [], []
        for i,track in enumerate(tqdm(allTracks,desc='Debug Initalisation: ',leave=True,position=0,delay=0.5)):
            IDS = []
            for t in range(params.t_init):
                ID = np.argwhere(np.linalg.norm( np.asarray(track)[t,1:4:] - Pt[t][:,1:4:] , axis=1) < params.d)[:,0]
                IDS.append(ID)
            if all(len(ID)>0 for ID in IDS):
                if np.std([ID[0] for ID in IDS])==0:
                    count+=1
                    Is.append(i)
                    Js.append(IDS[0][0])
        print('')
        print(len(Is),len(np.unique(Is)))
        print(len(Js),len(np.unique(Js)))
        tracks = [i for j, i in enumerate(tracks) if j not in Is]
        tracksj = [i for j, i in enumerate(tracksj) if j not in Js]
        print(str(count) + ' / ' + str(len(allTracks)) + ' ( '+str(round(count/len(allTracks)*100,2))+' % ) correct tracks')
        print(str(len(tracksj)) + ' / ' + str(len(P_clouds_true[0])) + ' ( '+str(round(len(tracksj)/len(P_clouds_true[0])*100,2))+' % ) unfound tracks')
        print('')
        print(len(tracks))
        # plot points
        '''
        fig, axis = plt.subplots(1,2,figsize=(16,8),subplot_kw=dict(projection='3d'))
        axis[0].set_title('ground truth')        
        for track in tqdm(tracks_true, desc='Plot tracks', position=0 , leave=True, delay=0.5):
            x, y, z = track[:,0], track[:,1], track[:,2]
            points = np.array([x,y,z]).transpose().reshape(-1,1,3)
            segs = np.concatenate([points[:-1],points[1:]],axis=1)
            lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel/2,params.maxvel/2),linewidths=0.4,alpha=1)
            lc.set_array(Init_Velocity3D(track)[:,2]) 
            axis[0].add_collection3d(lc)
        axis[1].set_title('initialised tracks')
        for track in tqdm(allTracks, desc='Plot tracks', position=0 , leave=True, delay=0.5):
            x, y, z = np.array(track[1])[:,0],np.array(track[1])[:,1],np.array(track[1])[:,2]
            points = np.array([x,y,z]).transpose().reshape(-1,1,3)
            segs = np.concatenate([points[:-1],points[1:]],axis=1)
            lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel/2,params.maxvel/2),linewidths=0.4,alpha=1)
            lc.set_array(np.array(track[2])[:,2]) 
            axis[1].add_collection3d(lc)
        plt.show()
        '''
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.set_xlim(0,1), axis.set_ylim(0,1), axis.set_zlim(0,1)
        for track in tqdm(tracks, desc='Plot tracks', position=0 , leave=True, delay=0.5):
            x, y, z = track[:,1],track[:,2],track[:,3]
            points = np.array([x,y,z]).transpose().reshape(-1,1,3)
            segs = np.concatenate([points[:-1],points[1:]],axis=1)
            lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel,params.maxvel),linewidths=0.4,alpha=1)
            lc.set_array(track[:,6]) 
            axis.add_collection3d(lc)
            #P1 = P_clouds_true[0][np.argmin(np.linalg.norm(P_clouds_true[0][:,1:4]-np.array(track[1])[0],axis=1)),1:4]
            #P2 = P_clouds_true[1][np.argmin(np.linalg.norm(P_clouds_true[1][:,1:4]-np.array(track[1])[1],axis=1)),1:4]
            #axis.scatter(P1[0],P1[1],P1[2],c='black')
            #axis.scatter(P2[0],P2[1],P2[2],c='black')
        plt.show()
    else:
        # plot results
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        for track in tqdm(allTracks, desc='Plot tracks', position=0 , leave=True, delay=0.5):
            x, y, z = np.array(track[1])[:,0],np.array(track[1])[:,1],np.array(track[1])[:,2]
            points = np.array([x,y,z]).transpose().reshape(-1,1,3)
            segs = np.concatenate([points[:-1],points[1:]],axis=1)
            lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel,params.maxvel),linewidths=0.4,alpha=1)
            lc.set_array(np.array(track[2])[:,2]) 
            axis.add_collection3d(lc)
        axis.set_xlim(0,300), axis.set_ylim(0,300), axis.set_zlim(0,300)
        plt.show()
if __name__ == "__main__":
    main()