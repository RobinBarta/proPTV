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
    case_name, runname, suffix, Zeros = '27000', 'run1', '_repaired', 5
    t_start, t_end, dt = 0, 29, 1
    loadBroken = False
    
    eps, misscounts = 0.0075, 3
    
# %%


def main(): 
    # load params
    params = Track_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    params.origin_path = params.case_name+'/analysis/origin/origin_{time}.txt'
    
    # get timeline
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    
    # load ground truth
    P_clouds_true = [np.loadtxt(params.origin_path.format(time=str(t).zfill(params.Zeros)),skiprows=1) for t in tqdm(times)]
    # tracks_true = [ np.asarray([[P_clouds_true[j][i,1],P_clouds_true[j][i,2],P_clouds_true[j][i,3]] for j in range(len(times))]) for i in range(len(P_clouds_true[0])) ]

    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in times:
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # debug
    pmp, pmt = np.zeros(len(times)), 0
    for i,track in enumerate(tqdm(allTracks,desc='PMP: ',leave=True,position=0,delay=0.5)):
        IDS = []
        for t in range(len(times)):
            T = np.argwhere(track[:,0]==t)[:,0]
            if len(T)>0:
                ID = np.argwhere(np.linalg.norm( track[T[0],1:4:] - P_clouds_true[t][:,1:4:] , axis=1) < params.eps)[:,0]
                if len(ID)>0:
                    IDS.append(ID[0])
                    pmp[t]+=1
                else:
                    IDS.append(np.nan)
        L = len(IDS)
        nancount = len(np.argwhere(np.isnan(IDS)==True)[:,0])
        IDS = np.asarray(IDS)[np.argwhere(np.isnan(IDS)==False)[:,0]]
        IDS = np.asarray(IDS,dtype=int)
        if len(IDS)>0:
            counts = np.bincount(IDS)
            IDtrue = np.argmax(counts)
            wrongcount = len(IDS[IDS!=IDtrue])
            count = wrongcount+nancount
            if count < int(np.rint(L/3)):
                pmt += 1
    pmp = np.round(pmp/len(P_clouds_true[0]) * 100,3)
    pmt = np.round(pmt/len(allTracks)*100,3)
    
    # plot 
    plt.rcParams.update({'font.family':'arial'})
    fig, axis = plt.subplots(figsize=(7,4),dpi=200)
    axis.spines['bottom'].set_linewidth(1.5), axis.spines['top'].set_linewidth(1.5)
    axis.spines['right'].set_linewidth(1.5), axis.spines['left'].set_linewidth(1.5)
    plt.grid(True)
    plt.title('pmt '+str(pmt)+' %', fontsize=8 , fontweight = 'bold')
    plt.xlabel('time', fontsize=8 , fontweight = 'bold')
    plt.ylabel('pmp [%]', fontsize=8 , fontweight = 'bold')
    plt.ylim(0,100)
    plt.plot(times,pmp,'o-',c='red')
    plt.tight_layout(), plt.show()
    
    # save tracks
    plt.savefig(params.case_name+'/analysis/allTracks'+params.suffix+'_pmp2.tif',dpi=200)
if __name__ == "__main__":
    main()  