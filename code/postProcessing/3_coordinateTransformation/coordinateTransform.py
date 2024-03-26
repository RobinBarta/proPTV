'''

    This Code performs a coordinate transform on the tracks coordinate system.
    
    To transform Lagrange velocities (vx) into Euler velocities (u) use the method: u = vx / dt.
    Here dt is the inverse recording frequency.
    
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
    case_name, runname, suffix = '27000', 'run1', '_origin'
    t_start, t_end = 0, 29
    cams = [0,1,2,3]
    loadBroken = False
    
    dt = 1 #Hz time unit 
    x_scale = 1 #[mm] position unit
    dx, dy, dz = 0, 0, 0 #[mm] shift
    
# %%

def main(): 
    # load params
    params = Units_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    print('Transform Coordinate System: ')
    allTracks_transform = []
    for track in tqdm(allTracks,desc=' conversions: ', leave=True,position=0,delay=0.5):
        ts = track[:,0]
        pos = np.array(track[:,1:4]) * params.x_scale
        pos[:,0] = pos[:,0] + params.dx
        pos[:,1] = pos[:,1] + params.dy
        pos[:,2] = pos[:,2] + params.dz
        vel = track[:,4:7] * params.x_scale / params.dt
        acc = track[:,7:10] * params.x_scale / params.dt**2
        xy = track[:,10::] 
        
        #vel = pos.copy()
        #vel[:,0] = signal.savgol_filter(pos[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode='interp') * params.x_scale / params.dt
        #vel[:,1] = signal.savgol_filter(pos[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode='interp') * params.x_scale / params.dt
        #vel[:,2] = signal.savgol_filter(pos[:,2],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode='interp') * params.x_scale / params.dt
        #acc = vel.copy()
        #xy = np.zeros([len(track),8])

        allTracks_transform.append([list(ts), list(pos), list(vel), list(acc), list(xy)])
        
    # save tracks
    SaveTracks(allTracks_transform, params, params.suffix+'_transformed' ,0 ,'w')
if __name__ == "__main__":
    main()  