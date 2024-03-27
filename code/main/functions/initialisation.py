'''

    This script contains general functions for the track initialisation routine.
    
'''


import joblib
import numpy as np

from tqdm import tqdm
from scipy import signal


def Initialisation(P_clouds,t,params):
    time = np.linspace(t-params.t_init+1,t,params.t_init,dtype=int)
    Ps = [P[:,:3:] for P in P_clouds]
    allTracks = []
    # for N_init run the initialisation process
    for i in range(params.N_init):
        maxVel = params.maxvel * (6-params.N_init+i+1)/(6)
        # initialse tracks forward in time
        tracks = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(Linking)(p0, Ps, maxVel, params) for p0 in tqdm(Ps[0],desc='   initialise: ',position=0,leave=True,delay=0.5))
        tracks_i = np.asarray([track for track in tracks if len(track)>0])
        if len(tracks_i)>0:
            tracks_i = UniqueFilter(tracks_i,params)
        # delete used points
        Ps, P_clouds, xy_used = ReducePoints(tracks_i,Ps,P_clouds,params)
        # put tracks in the correct allTracks format
        for j,track in tqdm(enumerate(tracks_i),desc='   generate tracks: ', position=0, leave=True, delay=0.5):
            allTracks.append([list(time),list(track[:,:3:]),list(track[:,3:6:]),list(track[:,6:9:]),[xy_used[k][j] for k in range(params.t_init)]])
    # process P_clouds and save out current Triag Points
    [np.savetxt(params.track_path+"currentTriagPoints_"+str(i-1)+".txt",P_clouds[i],delimiter=',',header="X,Y,Z,error,cx_i,cy_i") for i in range(1,params.t_init)]
    P_clouds = P_clouds[1::]
    P_clouds.append([])
    return allTracks, P_clouds

def Linking(p0,Ps,maxVel,params):
    # first link
    tracks = [np.array([p0,p1]) for p1 in FindNNPoints(p0, Ps[1], params.NN[0])]
    # second and ongoing links via polynom prediction
    if len(tracks)>0:
        # second and ongoing links via polynom prediction
        for i,P in enumerate(Ps[2::]):
            tracks = [[np.append(track,p.reshape(1,3),axis=0) for p in FindNNPoints((track+Init_Velocity3D(track))[-1],P,params.NN[i+1])] for track in tracks]
            tracks = np.concatenate(tracks)
        tracks = [np.hstack([track,Init_Velocity3D(track),Init_Acceleration3D(track)]) for track in tracks if np.max(np.linalg.norm(np.diff(track,axis=0),axis=1))<maxVel]
        if len(tracks)>0:
            # estimate best track by cost function based on minimal action principle
            E = np.linalg.norm( np.diff( [np.linalg.norm(track[:,3:6:],axis=1)**2 for track in tracks] ,axis=1) ,axis=1)
            track_final = tracks[np.argmin(E)]
            v_final = np.diff(track_final[:,:3:],axis=0)
            Theta_check = np.asarray([np.degrees(np.arccos(np.dot(v_final[i],v_final[i+1]) / (np.linalg.norm(v_final[i])*np.linalg.norm(v_final[i+1]))))<params.angle for i in range(len(v_final)-1)]).all()
            if Theta_check:
                return track_final
    return []

def FindNNPoints(p,P,N):
    # find N nearest neighbour points in P around p
    return P[np.argpartition(np.linalg.norm(p-P,axis=1), N)[:N:]]

def UniqueFilter(tracks,params):
    # only unique initalised tracks are passed
    IDs = []
    for t in tqdm(range(params.t_init),desc='   uniqueness filter: ', position=0, leave=True, delay=0.5):
        unique, counts = np.unique(tracks[:,t,0], return_counts=True)
        ID = np.argwhere(counts>1)[:,0]
        for i in ID:
            IDs.append(np.argwhere(tracks[:,t,0]==unique[i])[:,0])
    if len(IDs) != 0:
        IDs = np.unique(np.concatenate(IDs))
        tracks = [track for i,track in enumerate(tracks) if i not in IDs]
        return np.asarray(tracks)
    return np.asarray(tracks)

def ReducePoints(tracks,Ps,P_clouds,params):
    # reduce the Point clouds by the points occupied in tracks
    xy_used, Ps_new, P_clouds_new = [], [], []
    for i in tqdm(range(params.t_init),desc='   delete used 3D points: ', position=0, leave=True, delay=0.5):
        deleteList = [np.argwhere( np.linalg.norm(Ps[i]-track[i,:3:],axis=1)==0)[0][0] for track in tracks]
        xy_used.append( P_clouds[i][deleteList,4::] )
        Ps_new.append( np.delete(np.asarray(Ps[i]),np.asarray(deleteList),axis=0) if len(deleteList)!=0 else Ps[i] )
        P_clouds_new.append( np.delete(np.asarray(P_clouds[i]),np.asarray(deleteList),axis=0) if len(deleteList)!=0 else P_clouds[i] )
    return Ps_new, P_clouds_new, xy_used

def Init_Position3D(track):
    # estimate the position of a track of arbitory length
    savgol_mode = 'interp' #'nearest'
    pos = np.zeros_like(track)
    pos[:,0] = signal.savgol_filter(track[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=0,mode=savgol_mode)
    pos[:,1] = signal.savgol_filter(track[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=0,mode=savgol_mode)
    pos[:,2] = signal.savgol_filter(track[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=0,mode=savgol_mode)
    return pos

def Init_Velocity3D(track):
    # estimate the velocity of a track of arbitory length
    savgol_mode = 'interp' #'nearest'
    vel = np.zeros_like(track)
    vel[:,0] = signal.savgol_filter(track[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
    vel[:,1] = signal.savgol_filter(track[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
    vel[:,2] = signal.savgol_filter(track[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
    return vel

def Init_Acceleration3D(track):
    # estimate the acceleration of a track of arbitory length
    savgol_mode = 'interp' #'nearest'
    acc = np.zeros_like(track)
    acc[:,0] = signal.savgol_filter(track[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
    acc[:,1] = signal.savgol_filter(track[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
    acc[:,2] = signal.savgol_filter(track[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
    return acc
