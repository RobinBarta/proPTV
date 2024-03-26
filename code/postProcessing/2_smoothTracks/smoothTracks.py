'''

    This Code smooths all tracks with a savgol filter.
    
'''


import os
import numpy as np

from tqdm import tqdm
from scipy import signal

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Units_parameter():    
    case_name, runname = 'rbc_300mm_run4', 'proPTV_LSC_60_80'
    t_start, t_end = 0, 10
    cams = [0,1,2,3]
    loadBroken = False
    
# %%

def main(): 
    # load params
    params = Units_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    
    # load tracks
    allTracks = LoadTracks(params.track_path,'')
    if params.loadBroken == True:
        for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    savgol_mode = 'nearest' 
    allTracks_smoothed = []
    for track in tqdm(allTracks,desc='  smooth Tracks: ', leave=True,position=0,delay=0.5):
        ts = track[:,0]
        pos = track[:,1:4]
        pos[:,0] = signal.savgol_filter(pos[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),mode=savgol_mode)
        pos[:,1] = signal.savgol_filter(pos[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),mode=savgol_mode)
        pos[:,2] = signal.savgol_filter(pos[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),mode=savgol_mode)
        vel = track[:,4:7]
        #vel[:,0] = signal.savgol_filter(vel[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
        #vel[:,1] = signal.savgol_filter(vel[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
        #vel[:,2] = signal.savgol_filter(vel[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
        acc = track[:,7:10]
        #acc[:,0] = signal.savgol_filter(acc[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
        #acc[:,1] = signal.savgol_filter(acc[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
        #acc[:,2] = signal.savgol_filter(acc[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
        xy = track[:,10::] 
        allTracks_smoothed.append([list(ts), list(pos), list(vel), list(acc), list(xy)])
        
    # save tracks
    SaveTracks(allTracks_smoothed, params, '_smoothed' ,0 ,'w')
if __name__ == "__main__":
    main()  