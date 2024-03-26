'''

    This Code contains general functions for the linking procedure.
    Basically, triangulated 3D particle point clouds are linked to initial tracks

'''
# %%



# %%
import time , sys , joblib , itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

def Linking(P_clouds,params):
    '''
        Main Linking Code.
        Takes P_clouds and returns a linking List containing the indices of each 
        3D point cloud of which particles should be conntected to tracks.
    '''
    if all(len(value)>0 for value in P_clouds) == True:
        # creating the linking list , initialized with -1 
        links = -1*np.ones([len(P_clouds[0]),params.t_init],dtype=int)
        # the first row is filled with the IDs from the first Point Cloud
        links[:,0] = np.arange(len(P_clouds[0]))
        # go over all Point clouds and perform linking algorithm for each connection in P_clouds
        # e.g. for P_clouds = [P1,P2,P3] linking is performed from P1-P2 and P2-P3 and vice versa
        for ti in  np.arange(params.t_init-1) :
            P1 , P2 = P_clouds[ti] , P_clouds[ti+1]
            # initialize the sub linking list from P1 to P2 
            links12 = -1*np.ones([len(links[:,ti]),2],dtype=int)
            links12[:,0] = links[:,ti]
            # LINKING P1 TO P2
            links12[:,1] = np.asarray(joblib.Parallel(n_jobs=params.threads)(joblib.delayed(FindPartner)(_,P1,P2,params) 
                                                                             for _ in tqdm(links12[:,0])),dtype=int)
            # initialize the sub linking list from P2 to P1 
            links21 = -1*np.ones([len(P2),2],dtype=int)
            links21[:,0] = np.arange(len(P2))
            # LINKING P2 TO P1
            links21[:,1] = np.asarray(joblib.Parallel(n_jobs=params.threads)(joblib.delayed(FindPartner)(_,P2,P1,params) 
                                                                             for _ in tqdm(links21[:,0])),dtype=int)
            # only keep links where P1-P2 and P2-P1 link the same particles
            #links_BackToBack = np.asarray([link for link in links12])
            links_BackToBack = np.asarray([link for link in links12 
                                           if np.argwhere( (link[0] == links21[:,1]) & (link[1] == links21[:,0]) ).size == 1])
            # fill the next row with IDs from the next Point Cloud if there is a corresponding link ID
            if links_BackToBack.size>0:
                links[:,ti+1] = [links_BackToBack[np.argwhere(link[ti]==links_BackToBack[:,0])[0][0],1] 
                                 if np.argwhere(link[ti]==links_BackToBack[:,0]).size==1 else -1 for link in links] 
            # delete links where no partner is found
            links = np.asarray( links[links[:,ti+1]!=-1] , dtype=int)
        return links
    else:
        return np.empty([0,params.t_init],dtype=int)
    
def FindPartner(index , P1 , P2, params):
    '''
        Helper function from Linking.
        It searches for a linking partner in point cloud P2 for a given particle in point cloud P1.
        Based on histogram search.
    '''
    # get 3D Position p and cam intensies per camera
    p = P1[index,:3:]
    # create search radius around p with radius params.linkingHistRadius and collect all particles
    # inside this seach volume for P1 and P2
    possibleP1 = P1[np.argwhere(np.linalg.norm(p-P1[:,:3:],axis=1) < params.maxvel)[:,0] , :3:] 
    possibleP2_ids = np.argwhere( np.linalg.norm(p-P2[:,:3:],axis=1) < params.maxvel)
    # if possibleP2_ids is empty return -1 -> no link partner can be found
    if possibleP2_ids.size > 0:
        possibleP2 = P2[possibleP2_ids[:,0] , :3:]  
        # calculate the cartesian product from possibleP1 and possibleP2 and calculate all displacement vectors dP 
        dP = np.asarray([point[1]-point[0] for point in list(itertools.product(possibleP1, possibleP2))])  
        # calculate the mean displacement velocity from dP
        kdeVel = np.zeros(3)
        kdeVel[0] , kdeVel[1] , kdeVel[2] = KDE(dP[:,0]) , KDE(dP[:,1]) , KDE(dP[:,2])
        # calculate the expected particle position of the linking partner in P2
        p_new = p + kdeVel
        # check cost function if a suitable candidate of P2 for the linking is located near p_new 
        ds = np.linalg.norm(p_new-P2[:,:3:],axis=1)
        v_link = P2[np.argmin(ds),:3:]-p
        # check if the displacement velocity fullfills the tracking vel parameter, if yes return linking ID
        if np.linalg.norm(v_link)<params.maxvel:
            return np.argmin(ds)
    return -1

def KDE(dp):
    '''
        Helper function from FindPartner.
        Estimates the kdeVel from the linking histogram.
    '''
    if np.unique(dp).size >= 2:   
        bins = np.linspace(np.min(dp),np.max(dp),int(len(dp)/0.01))
        kde = stats.gaussian_kde(dp)
        kde.set_bandwidth(bw_method='silverman')
        return bins[np.argmax(kde(bins))]
    return np.mean(dp)


def GetTracks(P_clouds, t, links, params):
    '''
        transform the 3D point clouds at different time steps to tracks 
        by using the linking list stored in links.
    '''
    allTracks = []
    # for each track go over the link list
    for i in np.arange(len(links)):
        times , pos , vel , acc,  temperatures , camIs , camPs = [] , [] , [] , [] ,  [] , [] , []
        # for every particle linked to a track collect it from the P_clouds list
        for j in np.arange(params.t_init):
            times.append( t - ((params.t_init-1)*params.dt-(params.dt*j)) )
            pos.append( P_clouds[j][links[i,j],:3:] )
            camPs.append( P_clouds[j][links[i,j],4+len(params.cams)::] )
        vel = [pos[k+1]-pos[k] for k in np.arange(params.t_init-1)]
        vel.append(pos[-1]-pos[-2])
        acc = [np.zeros(3) for point in pos]
        
        ts = np.array(times).reshape(len(pos),1)
        pos = np.array(pos).reshape(len(pos),3)
        vel = np.array(vel).reshape(len(pos),3)
        acc = np.array(acc).reshape(len(pos),3)
        allTracks.append(np.hstack([ts,pos,vel,acc]))
    return allTracks
