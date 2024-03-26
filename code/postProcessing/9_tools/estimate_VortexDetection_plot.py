'''

    This Code detects vortex structures.
    
'''


import os, sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

os.chdir('../../../data')


# %%

class Detection_parameter():    
    case_name, runname, suffix, Zeros = 'rbc_300mm_run2', 'proPTV_reversal_20000_30000', 'Euler', 5
    
    t_start, t_end, dt = 29950, 30000, 50
    
    x0, x1, Nx = 0, 300, 32
    y0, y1, Ny = 0, 300, 32
    z0, z1, Nz = 0, 300, 32
    x0sub, x1sub, Nxsub = 0, 300, 32 # number of subvolumes in each direction
    y0sub, y1sub, Nysub = 0, 300, 32 # number of subvolumes in each direction
    z0sub, z1sub, Nzsub = 0, 300, 32 # number of subvolumes in each direction
    R = 300 # side length of square sub volume

# %%


def main():
    # load parameter
    params = Detection_parameter()
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/"
    
    # create timeline
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    
    # plot
    center = []
    print('Vortex detection:')
    for i in range(len(times)-1):
        t = times[i]
        print(' t = ' + str(t) + ' / ' + str(times[-1]))
        
        # load data
        data = np.loadtxt(params.field_path+params.suffix+'/'+params.suffix+'_t_{time}.txt'.format(time=str(t).zfill(params.Zeros)),skiprows=1)
        X, Y, Z, vx, vy, vz = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        vx /= np.sqrt(vx**2+vy**2+vz**2) / 5
        vy /= np.sqrt(vx**2+vy**2+vz**2) / 5
        vz /= np.sqrt(vx**2+vy**2+vz**2) / 5
        
        # load data
        gamma = np.loadtxt(params.field_path+"VortexDetection/gamma_"+str(t)+'.txt')
        x, y, z = np.meshgrid(np.linspace(params.x0sub,params.x1sub,params.Nxsub), np.linspace(params.y0sub,params.y1sub,params.Nysub), np.linspace(params.z0sub,params.z1sub,params.Nzsub))
        pos = np.vstack([x.ravel(),y.ravel(),z.ravel()]).T
        centerpos = pos[np.argmax(gamma)]
        
        # plot
        ax = plt.figure(figsize=(12,12)).add_subplot(111, projection='3d')
        ID = np.argwhere(gamma>0.8*np.max(np.abs(gamma)))[:,0]
        #ID = np.argwhere(np.abs(pos[:,1]-(params.x1-pos[:,0]))<50)[:,0]
        plot = ax.scatter(pos[ID,0], pos[ID,1], pos[ID,2], c=gamma[ID], cmap='seismic',norm=TwoSlopeNorm(np.nanmean(gamma)))
        #ax.quiver(X[ID], Y[ID], Z[ID], vx[ID], vy[ID], vz[ID], color='black')
        #ID = np.argwhere(np.abs(pos[:,1]-(150-pos[:,0]))<5)[:,0]
        #plot = ax.scatter(X, Y, Z, c=vz, cmap='seismic',norm=TwoSlopeNorm(0))
        ax.scatter(centerpos[0], centerpos[1], centerpos[2], c='black')
        #cbar = plt.colorbar(plot)
        #cbar.set_label(r'$\gamma$')
        ax.set_xlim(params.x0,params.x1), ax.set_ylim(params.y0,params.y1), ax.set_zlim(params.z0,params.z1)
        plt.tight_layout(), plt.show()
    
        #sys.exit()
if __name__ == "__main__":
    main()