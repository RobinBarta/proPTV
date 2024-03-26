'''

    This Code repairs tracks that broke apart during the main code.
    
'''


import os
import numpy as np

from trackrepair_functions import *

from functions.soloff import *
from functions.prediction import *
from functions.setup import *

os.chdir('../../data')


# %%

class Repair_parameter():    
    case_name, runname, suffix, Zeros = '27000', 'run1', '_backtracking', 5
    loadBroken, t_start, t_end = False, 0, 29
    cams = [0,1,2,3]
    
    dt_repair, maxvel = 1, 0.015 # for the moment dt_repair <= 2!
    Nmin = 3
    angle = 110
    
# %%

def main(): 
    # load params
    params = Repair_parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    
    # load tracks
    allTracks = LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/',params.suffix)
    if params.loadBroken == True:
        for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    allTracks = [track for track in allTracks if len(track[:,0]>=5)]
    print(' loaded ' + str(len(allTracks)) + ' tracks\n')
    
    # cut tracks by length 
    allTracks_torepair = [track for track in allTracks if len(track)>=params.Nmin]    
    allTracks_nottorepair = [track for track in allTracks if len(track)<params.Nmin]
    # perfrom repairing
    allTracks_repaired, IDs = RepairTracks(allTracks_torepair,params)    
    # delete used tracks
    allTracks_del = [track for i, track in enumerate(allTracks_torepair) if i not in IDs]
    # combine all tracks
    allTracks_final = allTracks_del + allTracks_repaired + allTracks_nottorepair
    print('\nrepaired ' + str(len(allTracks_torepair)-len(allTracks_del)) + ' tracks to ' + str(len(allTracks_repaired)) + ' tracks\n')
    # save tracks
    allTracks_final_new = []
    for track in allTracks_final: 
        ts, pos, vel, acc, xy = list(track[:,0]), list(track[:,1:4:]), list(track[:,4:7:]), list(track[:,7:10]), list(track[:,10:])
        allTracks_final_new.append([ts, pos, vel, acc, xy])
    SaveTracks(allTracks_final_new, params, '_repaired_all' ,0 ,'w')
    allTracks_final_repaired = []
    for track in allTracks_repaired: 
        ts, pos, vel, acc, xy = list(track[:,0]), list(track[:,1:4:]), list(track[:,4:7:]), list(track[:,7:10]), list(track[:,10:])
        allTracks_final_repaired.append([ts, pos, vel, acc, xy])
    SaveTracks(allTracks_final_repaired, params, '_repaired' ,0 ,'w')
if __name__ == "__main__":
    main()   