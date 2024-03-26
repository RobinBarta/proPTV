'''

=================================================================================================================
=	Name        :    proPTV     																		        =
=	Author      :    Robin Barta 			                                                                    =	
=	Version     :    1.0 (01.04.2024)																		    =
=   Copyright   :    DLR																						=
=================================================================================================================

    track format : [times, positions, velocities, acceleration, camPoints]

'''


import sys, joblib
import numpy as np

from datetime import datetime
from tqdm import tqdm

from config import *
sys.path.append('functions/')
from functions.setup import *
from functions.soloff import *
from functions.triangulation import *
from functions.initialisation import *
from functions.prediction import *
from functions.tracking import *
from functions.trackingBackwards import *

import warnings
warnings.filterwarnings("ignore")


def main(): 

    #%%     Setup phase
    
    # set clock
    start_time = datetime.today()
    # load parameter class from config.py
    params = Parameter()
    # create output folder
    params = SetupCase(params) 
    # get soloff parameter in the desired cam orientation
    ax , ay = GetSoloff(params)   
    
    print('================================================================')
    print('|========================= proPTV =============================|')
    print('|==============================================================|')
    print('\n using ' + str(joblib.cpu_count()) + ' threads\n')
    
    #%%     Initialisation phase

    print('|======================= INITIALISATION =======================|\n') 
    
    # check for loading initial tracks
    if params.loadOption == True:        
        # load inital tracks and 3D particles per time step
        times_init = np.linspace(params.t_start, params.t_start+params.t_init-1, params.t_init, dtype=int)[::params.dt]
        allTracks, P_clouds, lastTime = LoadInitalTracks(params)
        print(' loaded ' + str(len(allTracks)) + ' initial tracks at t = ' + str(lastTime) + '\n')
    else:
        # triangulate t_init point clouds from the particle lists of all cameras
        P_clouds = []
        # loop over the first params.t_start+params.dt*params.t_init time steps and triangulate particles
        times_init = np.linspace(params.t_start, params.t_start+params.dt*params.t_init-1, params.dt*params.t_init, dtype=int)[::params.dt]
        for t in times_init:
            print(' t = ' + str(t))
            # load particle lists [cx,cy] of the current time step for all cameras
            ImgPoints = [np.loadtxt(params.imgPoints_path.format(cam=cam,timeString=str(t).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
            print('  particles per cam: ' , str([len(imgP) for imgP in ImgPoints]), '\n')
            # triangulate 3D Point cloud from ImgPoints
            print('  Triangulation:')
            TriagPoints , ImgPoints = Triangulate3DPoints(ImgPoints,t,ax,ay,params)
            P_clouds.append(TriagPoints)
            print('  triangulated ' + str(len(TriagPoints)) + ' points' )
            print('  remaining particles per cam: ' , str([len(ImgP) for ImgP in ImgPoints]) , '\n\n')
        
        # initialise tracks based on P_clouds
        print('  triangulated particles per frame: ' , str([len(P) for P in P_clouds]), '\n\n')
        print('  Initialisation:')
        allTracks, P_clouds = Initialisation(P_clouds,times_init[-1],params)
        print('  initialised ' +  str(len(allTracks)) + ' tracks\n')
        # saving inital tracks to output folder
        SaveTracks(allTracks, params, "_initial", params.t_start, 'w')
        lastTime = times_init[-1]
        print('\n')
        
# %%            Tracking phase
    
    print('|========================= MAIN LOOP =========================|\n') 
    
    # calculate mainloop start time
    mainLoopStart = lastTime + params.dt
    if params.gaptracking == True and mainLoopStart != times_init[-1]+params.dt:
        mainLoopStart = lastTime
    if mainLoopStart >= params.t_end:
        print(' loaded last time step \n')
    
    # check for backtracking else go on with main loop
    if params.backtracking == True:
        print(' Backtracking:')
        allTracks = BackTracking(allTracks, ax, ay, params)  
        print('')
    else:
        times_main = np.linspace(mainLoopStart, params.t_end, params.t_end-mainLoopStart+1, dtype=int)[::params.dt]
        for t in times_main:
            print(' t = ' + str(t))
            # load particle lists [cx,cy] of the current and next time step for all cameras
            ImgPoints = [np.loadtxt(params.imgPoints_path.format(cam=cam,timeString=str(t).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
            if t<times_main[-1]:
                ImgPoints_gap = [np.loadtxt(params.imgPoints_path.format(cam=cam,timeString=str(t+params.dt).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
            else:
                ImgPoints_gap = [[] for cam in params.cams]
            print('  particles per cam: ' , str([len(imgP) for imgP in ImgPoints]))
            if params.gaptracking == True:
                print('  (gap) particles per cam: ' , str([len(imgP) for imgP in ImgPoints_gap]))
            # Extend , try to extend allTracks based on the predicted positions
            print('\n  Extend Tracks:')
            allTracks, extendCounter, extendgapCounter, brokenCounter, ImgPoints, allTracks_broken = Extend(allTracks, ImgPoints, ImgPoints_gap, t, ax, ay, params)
            print('  extended ' + str(extendCounter) + ' tracks')
            print('  extended ' + str(extendgapCounter) + ' tracks with gap')
            print('  broken tracks: ' + str(brokenCounter))
            # write out broken tracks
            SaveTracks(allTracks_broken, params, "_broken"+str(t), t , 'w')
            print('\n  remaining particles per cam: ' , str([len(imgP) for imgP in ImgPoints]) + '\n')
            # Triangulation , triangulate 3D Point cloud from reduced ImgPoints
            print('  Triangulation:')
            TriagPoints , ImgPoints = Triangulate3DPoints(ImgPoints,t,ax,ay,params)
            P_clouds[-1] = TriagPoints
            print('  triangulated ' + str(len(TriagPoints)) + ' points' )
            print('  remaining particles per cam: ' , str([len(ImgP) for ImgP in ImgPoints]) , '\n')
            # initialise tracks based on P_clouds
            print('  triangulated particles per frame: ' , str([len(P) for P in P_clouds]), '\n')
            print('  Initialisation:')
            newTracks, P_clouds = Initialisation(P_clouds,t,params)
            print('  initialised ' +  str(len(newTracks)) + ' new tracks\n\n')
            # append all new found Tracks to the allTracks list 
            allTracks += newTracks
        
# %%            Output phase            
       
    print('|========================== OUTPUT ==========================|\n')
     
    # save allTracks as .hdf5    
    if params.backtracking == True:
        SaveTracks(allTracks, params , "_backtracking" , params.t_end , 'a')
    else:
        SaveTracks(allTracks, params , "" , params.t_end , 'a')
    
    print('\n')
    print('|==============================================================|')
    print('|========================== FINISHED ==========================|')  
    print('================================================================')
    print('\n processing time: ' + str((datetime.today()-start_time).total_seconds()/60) + ' minutes')
    
if __name__ == "__main__":
    main()