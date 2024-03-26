'''

    This script contains general functions to setup a case and load or save tracks as .hdf5 file.

'''


import os, h5py, shutil
import numpy as np

from datetime import datetime
from tqdm import tqdm


def GetSoloff(params):
    # load camera calibration
    ax = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="x"),delimiter=',') for c in params.cams])
    ay = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="y"),delimiter=',') for c in params.cams])
    return ax , ay

def SetupCase(params):
    # add important paths to params
    params.case_path = "../../data/"+params.case_name+"/"   
    params.load_path = params.case_path + "output/" + params.load_name + "/tracks/"  
    params.calibration_path = params.case_path + "input/calibration/c{cam}/soloff_c{cam}{xy}.txt" 
    params.imgPoints_path = params.case_path + "input/particle_lists/c{cam}/c{cam}_{timeString}.txt"  
    # create output folder
    output_path = params.case_path + "output/" + params.output_name + datetime.today().strftime('_%d-%m-%Y_%H-%M-%S')
    os.mkdir(output_path), os.mkdir(output_path+"/tracks"), os.mkdir(output_path+"/triangulation"), os.mkdir(output_path+"/fields")
    params.track_path, params.triangulation_path = output_path + "/tracks/", output_path + "/triangulation/"
    # create output file for tracks
    output = h5py.File(params.track_path+"tracks.hdf5", 'w')
    output.close()
    # save current config file
    shutil.copy('config.py', output_path)
    return params

def SaveTracks(allTracks, params, name ,t ,mode):
    # save track list as hdf5 file
    saveFile = h5py.File(params.track_path+"tracks"+name+".hdf5", mode)
    for key, track in enumerate(tqdm(allTracks, desc='  saving tracks: ', position=0, leave=True, delay=0.5)):
        ts = np.array(track[0]).reshape(len(track[0]),1)
        pos = np.array(track[1]).reshape(len(track[1]),3)
        vel = np.array(track[2]).reshape(len(track[2]),3)
        acc = np.array(track[3]).reshape(len(track[3]),3)
        xy = np.ravel(track[4]).reshape(len(track[4]),int(2*len(params.cams)))
        datas = np.hstack([ts,pos,vel,acc,xy])
        saveFile.create_dataset("track_"+str(t)+"_"+str(key), datas.shape, dtype='float64', data=datas)
    saveFile.close()
    return 0

def LoadTracks(pathToTracks,name):
    # load tracks from hdf5 file
    data = h5py.File(pathToTracks+"tracks"+name+".hdf5", "r")
    tracks = [data[key][:] for key in tqdm(list(data.keys()),desc=' loading tracks: ',leave=True,position=0,delay=0.5)]
    data.close()
    return tracks

def LoadInitalTracks(params):
    # load the last t_init-1 particle clouds
    P_clouds = []
    for i in range(params.t_init-1):
        P_clouds.append(np.loadtxt(params.load_path+"currentTriagPoints_"+str(i)+".txt",delimiter=",",skiprows=1))
    P_clouds.append([])
    # load particle tracks
    allTracks, lastTime = [], 0 
    for track in LoadTracks(params.load_path,params.suffix):
        ts, pos, vel, acc, xy = list(track[:,0]), list(track[:,1:4:]), list(track[:,4:7:]), list(track[:,7:10]), list(track[:,10:])
        allTracks.append([ts, pos, vel, acc, xy])
        lastTime = ts[-1] if lastTime < ts[-1] else lastTime
    return allTracks, P_clouds, int(lastTime)

def LoadBrokenTracks(params,name):
    # load particle tracks
    allTracks = []
    for track in LoadTracks(params.load_path,name):
        ts, pos, vel, acc, xy = list(track[:,0]), list(track[:,1:4:]), list(track[:,4:7:]), list(track[:,7:10]), list(track[:,10:])
        allTracks.append([ts, pos, vel, acc, xy])
    return allTracks