'''

    This script compares the tracks with the origin data and estimated the percent of matched particles (pmp) per time step.
    
'''


import os, joblib, sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Track_parameter():    
    case_name, runname, suffix, Zeros = '36000_new', 'run1', '', 5
    t_start, t_end, dt = 0, 29, 1
    loadBroken = True
    
    eps, misscounts = 0.008, 3
    
# %%


def main(): 
    # load params
    params = Track_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    params.origin_path = params.case_name+'/analysis/origin/origin_{time}.txt'
    
    # get timeline
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    
    # load ground truth
    P_clouds_true = [np.loadtxt(params.origin_path.format(time=str(t).zfill(params.Zeros)),skiprows=1) for t in tqdm(times,position=0,leave=True)]
    tracks_true = [ np.asarray([[P_clouds_true[j][i,1],P_clouds_true[j][i,2],P_clouds_true[j][i,3]] for j in range(len(times))]) for i in range(len(P_clouds_true[0])) ]

    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in times:
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # make track format
    allTracks_new = np.empty((len(allTracks),30,3))
    for i,track in enumerate(tqdm(allTracks,position=0,leave=True)):
        t0, t1 = track[0,0], track[-1,0]
        N0, N1 = int(np.abs(params.t_start-t0)),  int(np.abs(params.t_end-t1))
        data0, data1 = np.nan*np.ones([N0,3]), np.nan*np.ones([N1,3])
        #if len(track)>3:
        track_new = track[:,1:4:]
        if len(data0)>0:
            track_new = np.append(data0,track_new,axis=0)
        if len(data1)>0:
            track_new = np.append(track_new,data1,axis=0)
        allTracks_new[i] = track_new
    
    # calculate metrics
    hit = 0
    a, b, c, d = [], [], [], []
    for i,track in enumerate(tqdm(tracks_true,position=0,leave=True)):
        ID = []
        for t in range(len(times)):
            dist = np.linalg.norm(track[t]-allTracks_new[:,t,:],axis=1)
            minID, mind = np.nanargmin(dist), np.nanmin(dist)
            if mind < params.eps:
                ID.append(minID)
        if len(ID)>0:
            hit += 1
            ID = np.unique(ID)
            F = len(ID)
            #C = np.sum([len(allTracks[ele]) for ele in ID])/30
            eps = [np.linalg.norm( allTracks_new[ele] - track, axis=1) for ele in ID]
            E = np.nanmean(eps)
            Cr = []
            C = 0
            for ele in range(len(ID)):
                epsi = eps[ele][np.isnan(eps[ele])==False]
                Args = np.argwhere(epsi<params.eps)
                Cr.append( len(Args)/len(epsi))
                C += len(Args)
            C /= 30
            a.append(F)
            b.append(C)
            c += Cr
            d.append(E)
            
            '''
            print('')
            print(F)
            print(C)
            print(E)
            print(Cr)
            
            fig = plt.figure(figsize=(8,7),dpi=200)
            axis = fig.add_subplot(111, projection='3d')
            axis.set_xlim(0,1), axis.set_ylim(0,1), axis.set_zlim(0,1)
            axis.scatter(track[0,0],track[0,1],track[0,2],c='black')
            axis.plot(track[:,0],track[:,1],track[:,2],c='black')
            for ele in ID:
                minTrack = allTracks[ele][:,1:4]
                axis.scatter(minTrack[0,0],minTrack[0,1],minTrack[0,2],c='red')
                axis.plot(minTrack[:,0],minTrack[:,1],minTrack[:,2],c='red')
            plt.show()
            #'''
    
    print('\n')
    print('hit tracks: ' + str(hit/len(tracks_true)) + ' %')
    print('F = ' + str(np.mean(a)))
    print('C = ' + str(np.mean(b)))
    print('Cr = ' + str(np.mean(c)))
    print('E = ' + str(np.mean(d)))
    
    string = 'with3'
    np.savetxt(params.case_name+'/analysis/pmp_a'+string+'.txt',a)
    np.savetxt(params.case_name+'/analysis/pmp_b'+string+'.txt',b)
    np.savetxt(params.case_name+'/analysis/pmp_c'+string+'.txt',c)
    np.savetxt(params.case_name+'/analysis/pmp_d'+string+'.txt',d)
    np.savetxt(params.case_name+'/analysis/pmp_hit'+string+'.txt',np.asarray([hit]))
if __name__ == "__main__":
    main()  