'''

    Script to estimate track length histogram.
    
'''


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import AutoMinorLocator

os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Track_parameter():    
    case_name, runname, suffix, Zeros = '27000', 'run1', '', 5
    t_start, t_end, dt = 0, 29, 1
    loadBroken = True
    
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
    
    # calculate track length histogram
    counts = [len(track[:,0]) for track in allTracks]
    values, counts = np.unique(counts, return_counts=True)
    
    # plot 
    Fontsize = 12
    Linewidth = 2
    Tickwidth = 2
    Pad = 5
    
    matplotlib.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(7,4),dpi=300)
    plt.grid(True)
    ax.set_xlim(-1,32)
    ax.set_ylim(-80,12000)
    ax.set_xticks([0,5,10,15,20,25,30],[0,5,10,15,20,25,30], fontsize=Fontsize-2,fontweight='bold')
    ax.set_yticks([0,2000,4000,6000,8000,10000,12000],[0,2000,4000,6000,8000,10000,12000], fontsize=Fontsize-2,fontweight='bold')
    ax.tick_params(axis='both', which='major', width=Tickwidth)
    ax.tick_params(axis='both', which='minor', width=Tickwidth)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel('track length', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    ax.set_ylabel('counts', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    
    ax.bar(values,counts,width=0.5,color='red',zorder=100)
    
    plt.tight_layout()
    plt.savefig(params.case_name+'/analysis/allTracks'+params.suffix+'_histogram.jpg',dpi=300)
    plt.show()
if __name__ == "__main__":
    main()  