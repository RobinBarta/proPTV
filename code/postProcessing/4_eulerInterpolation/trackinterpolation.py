'''

    This Code interpolates tracks.hdf5 to an Euler grid and an Lagrange Grid.
    
'''


import os
import numpy as np

from tqdm import tqdm

from trackinterpolation_functions import *
os.chdir('../../main')
from functions.setup import *

os.chdir('../../data')


# %%

class Interpolation_parameter():    
    case_name, runname, suffix = '27000', 'run1', ''
    t_start, t_end, dt = 10, 10, 1
    loadBroken, t0, t1 = False, 0, 29
    
    interpolationMode, smooth = 'nd', 0.0005 # rbf or nd
    dN = 1
    x0, x1, Nx = 0, 1, 50
    y0, y1, Ny = 0, 1, 50
    z0, z1, Nz = 0, 1, 50

# %%


def main():
    # load parameter
    params = Interpolation_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/"
    
    # create output folders
    os.makedirs(params.field_path+"Euler",exist_ok=True), os.makedirs(params.field_path+"Lagrange",exist_ok=True)
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in tqdm(np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int),leave=True,position=0,desc=' loading broken tracks'):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # interpolate tracks 
    print('Interpolation:')
    for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]:
        print(' t = ' + str(t))
        InterpolateTracksToGrid(t,allTracks,params) 
if __name__ == "__main__":
    main()