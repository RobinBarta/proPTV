
'''

    Plot script for tracks.
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import signal

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Track_parameter():    
    case_name, runname, suffix, ending = '9000', 'run1', '', ''
    t_start, t_end, dt = 0, 29, 1
    loadBroken = False
    
    x0, x1 = 0, 1
    y0, y1 = 0, 1
    z0, z1 = 0, 1
    
    # plot property
    timesep = 1
    maxvel = 0.015
    cplot = 2 # [0,1,2] = [u,v,w]
    # viewing angle
    a1, a2 = 21, 133
    
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
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # plot tracks
    Fontsize = 12
    Linewidth = 2
    Trackwidth = 0.5
    
    # settings
    fig = plt.figure(figsize=(8,7),dpi=200)
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('X [mm]',fontsize=Fontsize), axis.set_ylabel('Y [mm]',fontsize=Fontsize), axis.set_zlabel('Z [mm]',fontsize=Fontsize)
    axis.set_xlim(params.x0,params.x1), axis.set_ylim(params.y0,params.y1), axis.set_zlim(params.z0,params.z1)
    #axis.set_xticks([0,100,200,300],[0,100,200,300],fontsize=Fontsize)
    #axis.set_yticks([0,100,200,300],[0,100,200,300],fontsize=Fontsize)
    #axis.set_zticks([0,100,200,300],[0,100,200,300],fontsize=Fontsize)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.zaxis.set_minor_locator(AutoMinorLocator(2))
    axis.view_init(elev=params.a1, azim=params.a2)
    
    # plot 
    for track in tqdm(allTracks, desc='Plot tracks', position=0 , leave=True, delay=0.5):
        x, y, z = track[:,1], track[:,2], track[:,3]
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs,cmap='seismic',norm=Normalize(-params.maxvel,params.maxvel),linewidths=0.4,alpha=1)
        #lc.set_array(track[:,4+params.cplot]) 
        lc.set_array(signal.savgol_filter(z,window_length=min(len(z),5),polyorder=min([len(z)-1,3]),deriv=1,mode='interp')*params.timesep) 
        axis.add_collection3d(lc)
        
    #cbar = fig.colorbar(lc,orientation='horizontal')
    #cbar.set_label(r'$V_Z$ [mm/s]',fontsize=Fontsize)
    #cbar.set_ticks([-30,-20,-10,0,10,20,30],labels=['-30','-20','-10','0','10','20','30'])
    #cbar.ax.tick_params(labelsize=Fontsize-2)
    #cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # draw box
    '''
    axis.plot3D([params.x0,params.x1],[params.y0,params.y0],[params.z0,params.z0], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x1],[params.y1,params.y1],[params.z0,params.z0], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x1],[params.y1,params.y1],[params.z1,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x1],[params.y0,params.y0],[params.z1,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x0],[params.y0,params.y1],[params.z0,params.z0], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x1,params.x1],[params.y0,params.y1],[params.z0,params.z0], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x1,params.x1],[params.y0,params.y1],[params.z1,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x0],[params.y0,params.y1],[params.z1,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x0],[params.y0,params.y0],[params.z0,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x1,params.x1],[params.y0,params.y0],[params.z0,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x1,params.x1],[params.y1,params.y1],[params.z0,params.z1], color="black", linewidth=Linewidth, zorder=100)
    axis.plot3D([params.x0,params.x0],[params.y1,params.y1],[params.z0,params.z1], color="black", linewidth=Linewidth, zorder=100)
    '''
    
    # save 
    plt.savefig(params.case_name+'/analysis/allTracks'+params.suffix+'_'+params.ending+'.png',dpi=200)
    plt.show()
    
if __name__ == "__main__":
    main()  