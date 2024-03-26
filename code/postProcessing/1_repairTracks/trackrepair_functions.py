'''

    This script contains general functions for the track repair.
    
'''


import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from tqdm import tqdm
from scipy.spatial import KDTree

os.chdir('../../main')
from functions.prediction import *
from functions.initialisation import *

def Combine(lists):
    # combine lists uniquly in input order
    return [lists[i] for i in sorted(np.unique(lists, return_index=True)[1])]

def CheckIfAIsInB(A, B):
    # check if A is in B
    return any(A == B[i:i + len(A)] for i in range(len(B)-len(A) + 1))

def EstimateMostProbableConnection(track_time,track_pos): 
    track_vel, track_acc = track_pos.copy(), track_pos.copy()
    track_pos[:,0] = signal.savgol_filter(track_pos[:,0],window_length=5,polyorder=3,deriv=0,mode='interp')
    track_pos[:,1] = signal.savgol_filter(track_pos[:,1],window_length=5,polyorder=3,deriv=0,mode='interp')
    track_pos[:,2] = signal.savgol_filter(track_pos[:,2],window_length=5,polyorder=3,deriv=0,mode='interp')
    track_vel[:,0] = signal.savgol_filter(track_pos[:,0],window_length=5,polyorder=3,deriv=1,mode='interp')
    track_vel[:,1] = signal.savgol_filter(track_pos[:,1],window_length=5,polyorder=3,deriv=1,mode='interp')
    track_vel[:,2] = signal.savgol_filter(track_pos[:,2],window_length=5,polyorder=3,deriv=1,mode='interp')
    track_acc[:,0] = signal.savgol_filter(track_pos[:,0],window_length=5,polyorder=3,deriv=2,mode='interp')
    track_acc[:,1] = signal.savgol_filter(track_pos[:,1],window_length=5,polyorder=3,deriv=2,mode='interp')
    track_acc[:,2] = signal.savgol_filter(track_pos[:,2],window_length=5,polyorder=3,deriv=2,mode='interp')            
    P_predict, sigma_X, sigma_V, sigma_A, mean_X, mean_V, mean_A = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(3):
        # estimate most probable track pos, vel, acc with errors, return also basis function psi such as weights
        X, X_err, V, V_err, A, A_err, psi, mean_w, N_gmm = MostProbableTrack(track_pos[:,i],track_vel[:,i],track_acc[:,i],3)
        # predict track with most probable track
        X_pred,V_pred,A_pred = PredictGMM(X,V,A,mean_w,psi,N_gmm,3)
        P_predict[i], sigma_X[i], sigma_V[i], sigma_A[i], mean_X[i], mean_V[i], mean_A[i] = X_pred, X_err[-1], V_err[-1], A_err[-1], X[-1], V[-1], A[-1]
        #Plot_GMM(track_time,track_pos[:,i],track_vel[:,i],track_acc[:,i],X,X_err,V,V_err,A,A_err,X_pred,V_pred,A_pred)
    return track_pos[-1], P_predict, P_predict+V_pred , sigma_X

def RepairTracks(allTracks,params):
    allTracks_t0 = np.asarray([track[0,0] for track in allTracks])
    allTracks_t1 = np.asarray([track[-1,0] for track in allTracks])
    allTracks_P0 = np.asarray([track[0,1:4] for track in allTracks])
    allTracks_P1 = np.asarray([track[-1,1:4] for track in allTracks])
    
    ID_repair = []
    for i in tqdm(range(len(allTracks)),desc='Get repair list:',position=0,leave=True,delay=0.5):
        ID_rep = []
        # repair backwards
        ID_back = np.argwhere( (np.linalg.norm(allTracks_P0[i]-allTracks_P1,axis=1)<params.dt_repair*params.maxvel) & ((allTracks_t0[i]-allTracks_t1)<=params.dt_repair) & ((allTracks_t0[i]-allTracks_t1)>0) )[:,0]
        for ID in ID_back:
            # Prob Approx
            time, posi = allTracks[i][:,0], allTracks[i][::-1,1:4]            
            w, psi_X, psi_V, psi_A = GMM(time, posi)
            X1, V1, A1 = Approximate(time, w, psi_X, psi_V, psi_A)
            Xi, Xi2 = X1[-1] + V1[-1], (X1[-1]+V1[-1]) + (V1[-1]+A1[-1])
            std_Xi = np.max(np.linalg.norm(np.diff(X1,axis=0)))
            # Prob Approx
            time, posj = allTracks[ID][:,0], allTracks[ID][:,1:4]  
            w, psi_X, psi_V, psi_A = GMM(time, posj)
            X2, V2, A2 = Approximate(time, w, psi_X, psi_V, psi_A)
            Xj, Xj2 = X2[-1] + V2[-1], (X2[-1]+V2[-1]) + (V2[-1]+A2[-1])
            std_Xj = np.max(np.linalg.norm(np.diff(X2,axis=0)))
            Theta = np.degrees( np.arccos(np.dot(Xi-X1[-1],Xj-X2[-1])/(np.linalg.norm(Xi-X1[-1])*np.linalg.norm(Xj-X2[-1]))) )
            # check
            if (np.linalg.norm(Xi-Xj)<np.min([std_Xi,std_Xj])) & (Theta>params.angle):
                ID_rep.append([ID,i])
            if (np.linalg.norm(Xi-Xj)<np.min([std_Xi,std_Xj])) & (np.linalg.norm(Xi2-Xj2)<np.min([std_Xi,std_Xj])) & (Theta>params.angle) & (params.dt_repair==2):
                ID_rep.append([ID,i])
        # repair forward
        ID_for = np.argwhere( (np.linalg.norm(allTracks_P1[i]-allTracks_P0,axis=1)<params.dt_repair*params.maxvel) & ((allTracks_t0-allTracks_t1[i])<=params.dt_repair) & ((allTracks_t0-allTracks_t1[i])>0) )[:,0]
        for ID in ID_for:
            # Prob Approx
            time, posi = allTracks[i][:,0], allTracks[i][:,1:4]            
            w, psi_X, psi_V, psi_A = GMM(time, posi)
            X1, V1, A1 = Approximate(time, w, psi_X, psi_V, psi_A)
            Xi, Xi2 = X1[-1] + V1[-1], (X1[-1]+V1[-1]) + (V1[-1]+A1[-1])
            std_Xi = np.max(np.linalg.norm(np.diff(X1,axis=0)))
            # Prob Approx
            time, posj = allTracks[ID_for[0]][:,0], allTracks[ID_for[0]][::-1,1:4]  
            w, psi_X, psi_V, psi_A = GMM(time, posj)
            X2, V2, A2 = Approximate(time, w, psi_X, psi_V, psi_A)
            Xj, Xj2 = X2[-1] + V2[-1], (X2[-1]+V2[-1]) + (V2[-1]+A2[-1])
            std_Xj = np.max(np.linalg.norm(np.diff(X2,axis=0)))
            Theta = np.degrees( np.arccos(np.dot(Xi-X1[-1],Xj-X2[-1])/(np.linalg.norm(Xi-X1[-1])*np.linalg.norm(Xj-X2[-1]))) )
            # check
            if (np.linalg.norm(Xi-Xj)<np.min([std_Xi,std_Xj])) & (Theta>params.angle):
                ID_rep.append([i,ID])
            if (np.linalg.norm(Xi-Xj)<np.min([std_Xi,std_Xj])) & (np.linalg.norm(Xi2-Xj2)<np.min([std_Xi,std_Xj])) & (Theta>params.angle) & (params.dt_repair==2):
                ID_rep.append([i,ID])
        # append to repair list
        for ID_final in ID_rep:
            ID_repair.append(ID_final)
    # get unique list
    ID_repair = np.unique(ID_repair,axis=0)
    # merge repair list
    ID_lists_merged = []
    for i,IDs_i in enumerate(tqdm(ID_repair,desc='Combine repair list:',position=0,leave=True,delay=0.5)):
        IDs_merged = IDs_i.copy()
        for IDs_j in ID_repair[i::]:
            # merge forward
            if IDs_j[0] == IDs_merged[-1]:
                IDs_merged = np.append(IDs_merged,IDs_j)
            # merge backward
            elif IDs_j[-1] == IDs_merged[0]: 
                IDs_merged = np.append(IDs_j,IDs_merged)
        # combine IDs uniquly in order 
        IDs_merged = Combine(IDs_merged)
        # check if merged sequence is already taken into account
        Check = [True for ids in ID_lists_merged if CheckIfAIsInB(IDs_merged,ids) == True]
        if len(Check)==0:
            ID_lists_merged.append(IDs_merged)  
    # repair Tracks
    allTracks_repaired = []
    for ID in tqdm(ID_lists_merged,desc='Repair tracks:' , leave=True, position=0):
        track_repaired = allTracks[ID[0]]
        for i in np.arange(1,len(ID)):
            t0, t1 = track_repaired[-1,0], allTracks[ID[i]][0,0]
            if (t1-t0)==2:
                intersectionPoints = np.zeros([1,1+3+3+3+2*len(params.cams)])
                intersectionPoints[0,0] = t0 + 1
                intersectionPoints[0,1:4] = (track_repaired[-1,1:4:]+allTracks[ID[i]][0,1:4])/2
                intersectionPoints[0,10::] = np.nan
                track_repaired = np.append(track_repaired,intersectionPoints,axis=0)
            track_repaired = np.append(track_repaired,allTracks[ID[i]],axis=0)
        # correct track position , velocity and acceleration
        track_repaired[:,1:4] = Init_Position3D(track_repaired[:,1:4])
        track_repaired[:,4:7] = Init_Velocity3D(track_repaired[:,1:4])
        track_repaired[:,7:10] = Init_Acceleration3D(track_repaired[:,1:4])
        allTracks_repaired.append(track_repaired)
    try:
        return allTracks_repaired, np.concatenate(ID_lists_merged)
    except:
        return allTracks_repaired , []