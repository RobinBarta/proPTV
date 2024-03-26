import os, sys, cv2, random
import numpy as np

from tqdm import tqdm
from makeData_functions import *

#%%
class Parameter():
        # selected number of particles (total number of particles = 64000) 
    N_particles = 45000
        
    # cameras used
    cams = [0,1,2,3]
    # resolution of the camera images
    x_res , y_res, bit = 800, 800, 16
    # particle intensity parameters of the images (I0,I1 absolute values, dI in percent)
    I0, I1, dI, SNR = 20000, 22000, 5, 7
    # select time sample: start time, number of time steps, frequency
    t0 , Nt, freq = 540000, 30, 25
    # survival probability in [%]
    Prob_survive = 100   
    # number of marker planes
    N_planes = 6
    # number of markers per direction
    N_marker = 19
    # data paths
    zFill = 5
    path_calib = 'calibration/c{cam}/soloff_c{cam}{xy}.txt'
    path_input = 'input/PARTICLE_{time}.nc'
    path_output = 'output/{Number}_{times}'
    path_output_img = path_output+'/particle_images/c{cam}/c{cam}_{time}.tif'
    path_output_marker = path_output+'/calibration_images/c{cam}_marker/calib_c{cam}_{plane}_{time}.tif'
    path_output_origin = path_output+'/origin/origin_{time}.txt'
#%% 

def main():
    # load config parameter
    params = Parameter()
    # load calibration parameters
    ax, ay = [np.loadtxt(params.path_calib.format(cam=c,xy="x")) for c in params.cams], [np.loadtxt(params.path_calib.format(cam=c,xy="y")) for c in params.cams]
    
    # make output folder
    os.makedirs('output',exist_ok=True)
    os.makedirs(params.path_output.format(Number=int(params.N_particles),times=int(params.freq)),exist_ok=True)
    os.makedirs(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/particle_images',exist_ok=True) 
    os.makedirs(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/calibration_images',exist_ok=True) 
    os.makedirs(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/origin',exist_ok=True) 
    [os.makedirs(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/particle_images/c{cam}'.format(cam=c),exist_ok=True) for c in params.cams]
    [os.makedirs(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/calibration_images/c{cam}_marker'.format(cam=c),exist_ok=True) for c in params.cams]
    
    # create figure of the camera placement
    CameraPlacement(params)
    
    # generate marker list
    [GenerateMarker(cam,ax[i],ay[i],params) for i, cam in enumerate(params.cams)]
    
    # create list of selected time frames
    Frames = np.linspace(params.t0,params.t0+(params.freq*60*params.Nt)-params.freq*60,params.Nt,dtype=int)
    # create list of N out of N_particles particles and store the ID values 
    List = np.asarray(random.sample(range(64000),params.N_particles),dtype=int)
    # create an intensity value for each particle ID in List 
    Intensities = np.random.uniform(params.I0,params.I1,params.N_particles)
    
    # for each time step and generate the seeding particle image    
    for i, t in enumerate( tqdm(Frames, leave=True, position=0, delay=0.1,desc='Creating Images: ') ):
        # load particle data at time step t
        ti, x, y, z, u, v, w, T, p, Ra, Pr = LoadNETCDF(t,List,params)
        campositions = [ np.vstack([Soloff(x,y,z,ax[ci]),Soloff(x,y,z,ay[ci])]).T for ci in params.cams ]
        # for every camera create the PTV image of the current time step
        for j, cam in enumerate(params.cams):
            # for each particle create a flag if it is displayed on the images at the current time step with a given survival probability
            survivers = np.asarray([ True if (100-np.random.uniform(0,100))<=params.Prob_survive else False for pos in range(params.N_particles) ])
            # create a pixel coordinate list for all survived particles with their corresponding intensity value
            particles = np.vstack([ Soloff(x[survivers],y[survivers],z[survivers],ax[j]), Soloff(x[survivers],y[survivers],z[survivers],ay[j]), Intensities[survivers] ]).T
            # generate and save image for a given camera and a given time step
            image = GenerateImage(particles,params)   
            cv2.imwrite(params.path_output_img.format(cam=cam,time=str(i).zfill(params.zFill),Number=int(params.N_particles),times=int(params.freq)),image)    
        # write out ground truth information
        data_output = np.vstack([List,x,y,z,u,v,w,T,p]).T
        for ele in campositions:
            data_output = np.append(data_output,ele,axis=1)
        np.savetxt(params.path_output_origin.format(time=str(i).zfill(params.zFill),Number=int(params.N_particles),times=int(params.freq)), data_output, header='ID, X, Y, Z, U, V, W, T, P, xc0, yc0, xc1, yc1, xc2, yc2, xc3, yc3')
    # calculate the physical time shift between each frames and save info file
    dt = LoadNETCDF(Frames[1],0,params)[0]-LoadNETCDF(Frames[0],0,params)[0]
    np.savetxt(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/info.txt',[Frames[0],Frames[-1],Ra,Pr,dt,params.N_particles,params.Nt,params.freq,params.Prob_survive],header='t_start, t_end, Ra, Pr, dt, Nparticles, Ntimesteps, freq, survProb')

    # build animation of camera images
    fps = 5
    [Animation(c,fps,Frames,params) for c in params.cams]
    
    # build tracks
    BuildTracks(Frames,params)
if __name__ == "__main__":
    main()
