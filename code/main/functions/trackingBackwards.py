'''
    
    This script contains the track extension functions for backtracking.

'''


import joblib, itertools, os
import numpy as np

from tqdm import tqdm
from scipy import signal

from functions.setup import *
from functions.soloff import *
from functions.initialisation import *
from functions.prediction import *
from functions.tracking import *


def BackTracking(allTracks,ax,ay,params):
    '''
        main function for backtracking
    '''
    timeline = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    # load all broken tracks    
    for t in tqdm(timeline,desc='  load broken tracks: ',position=0,leave=True,delay=0.5):
        if os.path.isfile(params.load_path+'tracks_broken{time}.hdf5'.format(time=t)):
            allTracks += LoadBrokenTracks(params,'_broken{time}'.format(time=t))
    # perfrom the backtracking
    print('')
    for t in timeline[::-1]: 
        print('  t = ' + str(t))
        # split allTracks in tracks which can be tracked at time step t
        allTracks_toextend  = [track for track in allTracks if int(track[0][0]) == int(t+1)]
        allTracks_nottoextend = [track for track in allTracks if int(track[0][0]) != int(t+1)]
        # get all ImgPoints of the current time step
        ImgPoints = [np.loadtxt(params.imgPoints_path.format(cam=cam,timeString=str(t).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
        usedPoints = np.asarray([track[4][np.argwhere(track[0]==t)[0][0]] for track in allTracks_nottoextend if len(np.argwhere(track[0]==t))>0])
        if len(usedPoints)>0:
            for i in tqdm(range(len(params.cams)), desc='   delete used imgpoints by gap tracks', position=0,leave=True,delay=0.5):
                deleteList = [np.argmin(np.linalg.norm(ImgPoints[i]-usedP,axis=1)) for usedP in usedPoints[:,int(2*i):int(2*(i+1)):] if np.isnan(usedP[0])==False]
                ImgPoints[i] = np.delete(ImgPoints[i],deleteList,axis=0) if len(deleteList)>0 else ImgPoints[i]
        # extend Tracks backwards
        allTracks_extend = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(ExtendTrackBackwards)(track,ImgPoints,ax,ay,params) for track in tqdm(allTracks_toextend,desc='  extend: ',position=0,leave=True,delay=0.5))
        # get extend counter 
        extendCounter = len([track for track in allTracks_extend if int(track[0][0]) == int(t)])
        print('  extended ' + str(extendCounter) + ' / ' + str(len(allTracks_extend)) + ' tracks')
        # but all tracks together for the next time step
        allTracks = allTracks_nottoextend + allTracks_extend
        print('')
    return allTracks

def ExtendTrackBackwards(track, ImgPoints, ax, ay, params):
    # load track information
    time, pos = np.array(track[0])[:8:], np.array(track[1])[:8:][::-1]
    
    # Prob Approx
    w, psi_X, psi_V, psi_A = GMM(time, pos)
    # Approximate
    X, V, A = Approximate(time, w, psi_X, psi_V, psi_A)
    # Predict
    X_next = X[-1] + V[-1] # Init_Position3D(pos)[-1] + Init_Velocity3D(pos)[-1]
    # Uncertanty
    #positions = [posi for posi in positions if len(posi)==len(time)]
    #w_mean, std_X, std_V, std_A = Uncertanty(time, positions, X, V, A, w, psi_X, psi_V, psi_A, 3)
    #std_X_next = std_X[-1] + std_V[-1]
    # define search radius for candidates
    #r = np.asarray([np.nanmin([np.linalg.norm( np.asarray([ F(X_next+std_X_next,ax[ci]), F(X_next+std_X_next,ay[ci]) ]) - np.asarray([ F(X_next-std_X_next,ax[ci]), F(X_next-std_X_next,ay[ci]) ]) ),params.epsR]) for ci in range(len(params.cams))])
    r = np.asarray([params.epsR for ci in range(len(params.cams))])
    
    # search for candidates on images and create all permutations with active cams >= params.activeMatches_extend
    candidatesP = np.array([xy for xy in FindCamCandidates(ImgPoints,X_next,r,ax,ay,params) if len(np.argwhere(np.isnan(xy[:,0])==False)[:,0])>=params.activeMatches_extend])
    # search for the best candidate
    if len(candidatesP) > 0:
        # triangulate candidates
        Candidates = np.array([NewtonSoloff_Extend(setP[np.argwhere(np.isnan(setP[:,0])==False)[:,0]], X_next, np.asarray(ax)[np.argwhere(np.isnan(setP[:,0])==False)[:,0]], np.asarray(ay)[np.argwhere(np.isnan(setP[:,0])==False)[:,0]]) for setP in candidatesP])
        #Prob = np.asarray([1-TrackingProbability(candi,X_next,std_X_next,3) for candi in Candidates[:,:3]])
        Prob = np.asarray([1])
        bestID = np.argmin((Candidates[:,3]/np.max(Candidates[:,3])) * (Prob/np.max(Prob)))
        # extend track if position lies inside measurment volume and velocity crit is fullfilled 
        if all((np.asarray(params.Vmin)<=Candidates[bestID,:3:]-np.asarray(params.Vmin))&(Candidates[bestID,:3:]-np.asarray(params.Vmin)<=np.asarray(params.Vmax)-np.asarray(params.Vmin))) and np.linalg.norm(Candidates[bestID,:3:]-pos[-1])<=params.maxvel:
            track[0] = [track[0][0]-params.dt] + track[0]
            track[1] = [Candidates[bestID,:3:]] + track[1]
            track[1] = list(Init_Position3D(np.array(track[1])))
            track[2] = list(Init_Velocity3D(np.array(track[1])))
            track[3] = list(Init_Acceleration3D(np.array(track[1])))
            track[4] = [np.ravel(candidatesP[bestID])] + track[4]
            return track
    return track