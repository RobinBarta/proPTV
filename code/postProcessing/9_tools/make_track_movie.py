'''

    This makes a movie of the tracks in windows.
    
'''


import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Units_parameter():    
    case_name, runname, suffix = '27000', 'run1', ''
    t_start, t_end, dt = 0, 25, 1
    cams = [0,1,2,3]
    loadBroken = True
    
    x0, x1, Nx = 0, 500, 32
    y0, y1, Ny = 0, 500, 32
    z0, z1, Nz = 0, 500, 32
    
    fps = 10
    window = 5
    maxvel = 0.015
    
# %%

def main(): 
    # load params
    params = Units_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    
    os.makedirs(params.case_name+"/analysis/track_movie_w_"+str(int(params.window)),exist_ok=True)
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    print('Make movie images: ')
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    times_window = np.linspace(times[0],times[-1]-params.window+1,len(times)-params.window+1,dtype=int)
    for t0 in times_window:
        print(' t = ' + str(t0))
        t1 = t0+params.window-1
        # plot tracks
        fig = plt.figure(figsize=(8,8))
        axis = fig.add_subplot(111, projection='3d')
        axis.view_init(21,80)
        for track in tqdm(allTracks, desc='  generate plot', position=0 , leave=True, delay=0.5):
            ks = np.argwhere((t0<=track[:,0]) & (track[:,0]<=t1))[:,0]
            if len(ks)>1:
                x, y, z = track[ks[0]:ks[-1]+1,1]*500, track[ks[0]:ks[-1]+1,2]*500, track[ks[0]:ks[-1]+1,3]*500
                points = np.array([x,y,z]).transpose().reshape(-1,1,3)
                segs = np.concatenate([points[:-1],points[1:]],axis=1)
                lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel,params.maxvel),linewidths=0.4,alpha=1)
                lc.set_array(track[ks[0]:ks[-1]+1,6]) 
                axis.add_collection3d(lc)
        axis.set_xlim(params.x0,params.x1)
        axis.set_ylim(params.y0,params.y1)
        axis.set_zlim(params.z0,params.z1)
        axis.set_xlabel('X [mm]'), axis.set_ylabel('Y [mm]'), axis.set_zlabel('Z [mm]')
        plt.tight_layout(), plt.show()
        # save tracks
        plt.savefig(params.case_name+'/analysis/track_movie_w_'+str(int(params.window))+'/tracks_'+str(t0)+'.png',dpi=200)
        plt.close('all')
    print('')
    
    # make avi
    height, width, layers = cv2.imread(params.case_name+'/analysis/track_movie_w_'+str(int(params.window))+'/tracks_'+str(times_window[0])+'.png').shape
    video = cv2.VideoWriter(params.case_name+'/analysis/track_movie_w_'+str(int(params.window))+'/movie.avi',cv2.VideoWriter_fourcc(*'XVID'),params.fps,(width,height))
    for t in tqdm(times_window, desc=' generate movie', position=0 , leave=True, delay=0.5):
        video.write(cv2.imread(params.case_name+'/analysis/track_movie_w_'+str(int(params.window))+'/tracks_'+str(t)+'.png'))
    cv2.destroyAllWindows()
    video.release()
if __name__ == "__main__":
    main()  