'''

    Script to estimate track overview.
    
'''


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import AutoMinorLocator

from tqdm import tqdm

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Track_parameter():    
    case_name, runname, suffix, Zeros = '27000', 'run1', '', 5
    t_start, t_end, dt = 0, 29, 1
    
# %%


def main(): 
    # load params
    params = Track_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    params.triag_path = params.case_name+'/output/'+params.runname+'/triangulation/'
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    # load broken tracks 
    for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
        if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
            allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # make overview
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    log = np.zeros([len(times),4])
    for t in tqdm(times,position=0,leave=True,desc='Calculate overview',delay=0.5): 
        triagParticles = 0
        if os.path.isfile(params.triag_path+'Points_{time}.txt'.format(time=str(t).zfill(params.Zeros))):
            triagP = np.loadtxt(params.triag_path+'Points_{time}.txt'.format(time=str(t).zfill(params.Zeros)),skiprows=1)
            log[t,0] = len(triagP)
        init, active = 0, 0
        for track in allTracks:
            ID = np.argwhere(track[:,0]==t)[:,0]
            if len(ID) > 0:
                active += 1
        log[t,2] = active
        if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
            broken = LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
            log[t,3] = len(broken)
    log[-3:,2] = log[-6:-3,2]
    log[0,1] = log[0,2]
    log[1::,1] = np.diff(log[:,2]) + log[1::,3]
    np.savetxt(params.case_name+'/analysis/log.txt',log)
    
    # plot 
    Fontsize = 12
    Linewidth = 2
    Tickwidth = 2
    Pad = 5
    
    matplotlib.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(7,4),dpi=300)
    plt.grid(True)
    ax.set_xlim(-1,31)
    ax.set_ylim(-1000,27000)
    ax.set_xticks([0,5,10,15,20,25,30],[0,5,10,15,20,25,30], fontsize=Fontsize-2,fontweight='bold')
    ax.set_yticks([0,9000,18000,27000],[0,9000,18000,27000], fontsize=Fontsize-2,fontweight='bold')
    ax.tick_params(axis='both', which='major', width=Tickwidth)
    ax.tick_params(axis='both', which='minor', width=Tickwidth)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel('time steps', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    ax.set_ylabel('counts', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    
    plt.plot( np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1) , log[:,0] ,'o-',c='green',label='triangulated particles',zorder=100)
    plt.plot( np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1) , log[:,1] ,'o-',c='orange',label='initialized tracks',zorder=100)
    plt.plot( np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1) , log[:,2] ,'o-',c='red',label='active tracks',zorder=100)
    plt.plot( np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1) , log[:,3] ,'o-',c='blue',label='broken tracks',zorder=100)
    
    ax.legend(loc='best', prop=font_manager.FontProperties(family='arial', weight='bold', size=Fontsize-3))
    plt.tight_layout()
    plt.savefig(params.case_name+'/analysis/allTracks'+params.suffix+'_overview.jpg',dpi=300)
    plt.show()
    
if __name__ == "__main__":
    main()  