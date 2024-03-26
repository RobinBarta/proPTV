'''
    
    This script contains the track extension functions.

'''


import joblib, itertools
import numpy as np

from tqdm import tqdm
from scipy import signal
from scipy.spatial import KDTree

from functions.setup import *
from functions.soloff import *
from functions.prediction import *
from functions.initialisation import *


def Extend(allTracks, ImgPoints, ImgPoints_gap, t, ax, ay, params):   
    ''' 
        Extending Tracks Main Function
    '''
    # check if track exist at the current time step and delete used img points
    allTracks_gapext = []
    if params.gaptracking == True:
        allTracks_gapext = [track for track in allTracks if int(track[0][-1]) == int(t)]
        allTracks = [track for track in allTracks if int(track[0][-1]) == int(t-params.dt)]
        usedPoints = np.asarray([track[4][-1] for track in allTracks_gapext])
        if len(usedPoints)>0:
            for i in tqdm(range(len(params.cams)), desc='   delete used imgpoints by gap tracks', position=0,leave=True,delay=0.5):
                deleteList = [np.argmin(np.linalg.norm(ImgPoints[i]-usedP,axis=1)) for usedP in usedPoints[:,int(2*i):int(2*(i+1)):] if np.isnan(usedP[0])==False]
                ImgPoints[i] = np.delete(ImgPoints[i],deleteList,axis=0) if len(deleteList)>0 else ImgPoints[i]
    # extend allTracks
    allTracks_past = [[track[0][:-8],track[1][:-8],track[2][:-8],track[3][:-8],track[4][:-8]] for track in allTracks]
    allTracks_future = [[track[0][-8:],track[1][-8:],track[2][-8:],track[3][-8:],track[4][-8:]] for track in allTracks]
    # build KD tree from the last track positions
    Ps = np.asarray([track[1][-1] for track in allTracks_future])
    KD = KDTree(Ps)
    ID_KD = [KD.query_ball_point(track[1][-1],1.5*np.max(np.linalg.norm(np.diff(track[1],axis=0),axis=1))) for track in allTracks_future] 
    allTracks_KD = [ [np.asarray(allTracks_future[ele][1]) for ele in ID] for ID in ID_KD]

    #allTracks_future_extended = [ExtendTrack(track,allTracks_KD[i],ImgPoints,ImgPoints_gap,ax,ay,params) for i,track in enumerate(tqdm(allTracks_future,delay=0.5,position=0,leave=True,desc='  extend: '))]
    allTracks_future_extended = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(ExtendTrack)(track,allTracks_KD[i],ImgPoints,ImgPoints_gap,ax,ay,params) for i,track in enumerate(tqdm(allTracks_future,delay=0.5,position=0,leave=True,desc='   extend: ')))
    
    allTracks_extended = [[track0[0]+track1[0],track0[1]+track1[1],track0[2]+track1[2],track0[3]+track1[3],track0[4]+track1[4]] for track0,track1 in zip(allTracks_past,allTracks_future_extended)]
    # sort allTracks by broken tracks, extended and gap-extended tracks 
    allTracks_broken = [track for track in allTracks_extended if int(track[0][-1]) == int(t-params.dt)]
    allTracks_extend = [track for track in allTracks_extended if int(track[0][-1]) == int(t)]
    allTracks_gapext2 = [track for track in allTracks_extended if int(track[0][-1]) > int(t)]
    
    # Delete used cam points
    #ImgPoints_new = DeleteImgPointsAfterExtend(ImgPoints,allTracks_extend,t,params)
    usedPoints = np.asarray([track[4][-1] for track in allTracks_extend])
    ImgPoints_new = []
    for i in tqdm(range(len(params.cams)), desc='   delete used imgpoints ', position=0,leave=True,delay=0.5):
        deleteList = [np.argmin(np.linalg.norm(ImgPoints[i]-usedP,axis=1)) for usedP in usedPoints[:,int(2*i):int(2*(i+1)):] if np.isnan(usedP[0])==False]
        ImgPoints_new.append( np.delete(ImgPoints[i],deleteList,axis=0) if len(deleteList)>0 else ImgPoints[i] )
    
    # caluclate the extend counters
    extendCounter, extendgapCounter, brokenCounter = len(allTracks_extend), len(allTracks_gapext2), len(allTracks_broken)
    return allTracks_extend+allTracks_gapext+allTracks_gapext2, extendCounter, extendgapCounter, brokenCounter, ImgPoints_new, allTracks_broken

def ExtendTrack(track,positions,ImgPoints,ImgPoints_gap,ax,ay,params):
    # load track information
    time, pos, vel, acc = np.array(track[0]), np.array(track[1]), np.array(track[2]), np.array(track[3])
    
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
        if all(params.Vmin<Candidates[bestID,:3:]) and all(Candidates[bestID,:3:]<params.Vmax) and np.linalg.norm(Candidates[bestID,:3:]-pos[-1])<=params.maxvel:
            track[0].append(int(time[-1]+params.dt))
            track[1].append(Candidates[bestID,:3:])
            track[1] = list(Init_Position3D(np.array(track[1])))
            track[2] = list(Init_Velocity3D(np.array(track[1])))
            track[3] = list(Init_Acceleration3D(np.array(track[1])))
            track[4].append(np.ravel(candidatesP[bestID]))
            return track
    
    # try gap tracking if no candidates could be found
    if params.gaptracking == True: 
        time, pos, vel, acc = np.array(track[0]), np.array(track[1]), np.array(track[2]), np.array(track[3])
        # Prob Approx
        w, psi_X, psi_V, psi_A = GMM(time, pos)
        # Approximate
        X, V, A = Approximate(time, w, psi_X, psi_V, psi_A)
        # Predict
        X_next = (X[-1]+V[-1]) + (V[-1]+A[-1])
        # Uncertanty
        #positions = [posi for posi in positions if len(posi)==len(time)]
        #w_mean, std_X, std_V, std_A = Uncertanty(time, positions, X, V, A, w, psi_X, psi_V, psi_A, 3)
        #std_X_next = std_X[-1] + std_V[-1]
        # define search radius for candidates
        #r = 2*np.asarray([np.nanmin([np.linalg.norm( np.asarray([ F(X_next+std_X_next,ax[ci]), F(X_next+std_X_next,ay[ci]) ]) - np.asarray([ F(X_next-std_X_next,ax[ci]), F(X_next-std_X_next,ay[ci]) ]) ),params.epsR]) for ci in range(len(params.cams))])
        r = 2*np.asarray([params.epsR for ci in range(len(params.cams))])
        
        # search for candidates on images and create all permutations with active cams >= params.activeMatches_extend
        candidatesP = np.array([xy for xy in FindCamCandidates(ImgPoints_gap,X_next,r,ax,ay,params) if len(np.argwhere(np.isnan(xy[:,0])==False)[:,0])>=params.activeMatches_extend])
        # search for the best candidate
        if len(candidatesP) > 0:
            # triangulate candidates
            Candidates = np.array([NewtonSoloff_Extend(setP[np.argwhere(np.isnan(setP[:,0])==False)[:,0]], X_next, np.asarray(ax)[np.argwhere(np.isnan(setP[:,0])==False)[:,0]], np.asarray(ay)[np.argwhere(np.isnan(setP[:,0])==False)[:,0]]) for setP in candidatesP])
            #Prob = np.asarray([1-TrackingProbability(candi,X_next,std_X_next,3) for candi in Candidates[:,:3]])
            Prob = np.asarray([1])
            # estimate best extend candidate
            bestID = np.argmin((Candidates[:,3]/np.max(Candidates[:,3])) * (Prob/np.max(Prob)))
            # extend track if position lies inside measurment volume and velocity crit is fullfilled 
            if all(params.Vmin<Candidates[bestID,:3:]) and all(Candidates[bestID,:3:]<params.Vmax) and np.linalg.norm(Candidates[bestID,:3:]-pos[-1])<=2*params.maxvel:
                track[0].append(int(time[-1]+params.dt))
                track[0].append(int(time[-1]+2*params.dt))
                track[1].append((pos[-1]+Candidates[bestID,:3:])/2)
                track[1].append(Candidates[bestID,:3:])
                track[1] = list(Init_Position3D(np.array(track[1])))
                track[2] = list(Init_Velocity3D(np.array(track[1])))
                track[3] = list(Init_Acceleration3D(np.array(track[1])))
                track[4].append(np.ravel(candidatesP[bestID]))              
                return track
    return track

def FindCamCandidates(ImgPoints,pos_pred,r,ax,ay,params):
    # find the candidates
    candidates = [np.squeeze(ImgPoints[i][np.argwhere( np.linalg.norm( np.array([ F(pos_pred,ax[i]) , F(pos_pred,ay[i]) ]) - ImgPoints[i] , axis = 1 ) < r[i] )]) for i in range(len(params.cams))]
    candidates = [candis.reshape(int(candis.size/2),2)[:] if candis.size!=0 else np.nan*np.ones([1,2]) for candis in candidates]
    candidates = [candis if len(candis)==1 else np.append(candis,np.nan*np.ones([1,2]) ,axis=0) for candis in candidates]
    # permute the candidates
    candidates_permutations = np.asarray(list(itertools.product(*candidates))) 
    candidatesP_permutations = candidates_permutations
    return candidatesP_permutations

def DeleteImgPointsAfterExtend(ImgPoints,allTracks,t,params):
    # search each ID of each ImgPoints occupied by tracks
    ImgPoints_new = []
    for i in tqdm(range(len(params.cams)), desc='   delete used imgpoints ', position=0,leave=True,delay=0.5):
        deleteList = np.array([np.argwhere(np.linalg.norm(ImgPoints[i]-track[4][np.argwhere(np.asarray(track[0],dtype=int)==int(t))[0][0]][int(2*i):int(2*(i+1)):],axis=1)==0)[:,0] for track in allTracks if np.argwhere(np.asarray(track[0],dtype=int)==int(t)).size>0],dtype=object)
        deleteList = [ID[0] for ID in deleteList if len(ID)>0]
        ImgPoints_new.append( np.delete(ImgPoints[i],deleteList,axis=0) if len(deleteList)>0 else ImgPoints[i] )
    return ImgPoints_new