'''

    Calculate Frenet frame for a given track.
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Track_parameter():    
    case_name, runname, suffix = 'syn_8000_20', 'proPTV_8000_0_10', ''
    t_start, t_end, dt = 0, 10, 1
    loadBroken = True
    
    # track ID
    N, div = 3256, 1000
    # maxvel and plot property
    maxvel, cplot = 0.01, 2 # [0,1,2] = [u,v,w]
    
# %%


def main(): 
    # load params
    params = Track_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks')
    pos = allTracks[params.N][:,1:4]
    vel = allTracks[params.N][:,4+params.cplot]
    
    #t = np.linspace(0,1,10)
    #pos = np.vstack([np.sin(t),np.cos(t),t]).T
    
    # calculate frenet frame
    # Get the tangent vector
    tangent = np.diff(pos, axis=0)
    tangent = tangent / np.linalg.norm(tangent, axis=1)[:, np.newaxis]
    # Get the normal vector
    normal = np.cross(tangent[:-1], np.diff(tangent, axis=0))
    normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
    # Get the binormal vector
    binormal = np.cross(tangent[:-1], normal)
    binormal = binormal / np.linalg.norm(binormal, axis=1)[:, np.newaxis]
    tangent, normal, binormal = tangent/params.div, normal/params.div, binormal/params.div

    # plot frenet frame
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(111, projection='3d')
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    points = np.array([x,y,z]).transpose().reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel/2,params.maxvel/2),linewidths=0.4,alpha=1)
    lc.set_array(vel) 
    axis.add_collection3d(lc)
    for i in range(len(pos)-2):
        axis.plot([pos[i,0],pos[i,0]+tangent[i,0]], [pos[i,1],pos[i,1]+tangent[i,1]], [pos[i,2],pos[i,2]+tangent[i,2]], color="red")
        axis.plot([pos[i,0],pos[i,0]+normal[i,0]], [pos[i,1],pos[i,1]+normal[i,1]], [pos[i,2],pos[i,2]+normal[i,2]], color="green")
        axis.plot([pos[i,0],pos[i,0]+binormal[i,0]], [pos[i,1],pos[i,1]+binormal[i,1]], [pos[i,2],pos[i,2]+binormal[i,2]], color="blue")
    axis.set_xlim(np.min(pos[:,0]),np.max(pos[:,0]))
    axis.set_ylim(np.min(pos[:,1]),np.max(pos[:,1]))
    axis.set_zlim(np.min(pos[:,2]),np.max(pos[:,2]))
    plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()  