'''

    This Code detects vortex structures.
    
'''


import os, sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir('../../../data')


# %%

class Detection_parameter():    
    case_name, runname, suffix, Zeros = 'rbc_300mm_run2', 'proPTV_reversal_20000_30000', 'Euler', 5
    
    t_start, t_end, dt = 20000, 30000, 50
    
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
    
    # create output folders
    os.makedirs(params.field_path+"VortexDetection",exist_ok=True)
    
    # create timeline
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    
    # detect vortex
    center = []
    print('Vortex detection:')
    for i in range(len(times)-1):
        t = times[i]
        print(' t = ' + str(t) + ' / ' + str(times[-1]))
        # load data
        data = np.loadtxt(params.field_path+params.suffix+'/'+params.suffix+'_t_{time}.txt'.format(time=str(t).zfill(params.Zeros)),skiprows=1)
        times_mean = np.linspace(times[i]+1,times[i+1]-1,times[i+1]-times[i]-1,dtype=int)
        #for n in times_mean:
        #    data[3::] += np.loadtxt(params.field_path+params.suffix+'/'+params.suffix+'_t_{time}.txt'.format(time=str(n).zfill(params.Zeros)),skiprows=1)[3::]
        #data[3::] /= (len(times_mean)+1)
        x, y, z, vx, vy, vz = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        pos, vel = np.vstack([x.ravel(),y.ravel(),z.ravel()]).T, np.vstack([vx.ravel(),vy.ravel(),vz.ravel()]).T        
        # calculate subvolumes
        xi, yi, zi = np.meshgrid(np.linspace(params.x0sub,params.x1sub,params.Nxsub), np.linspace(params.y0sub,params.y1sub,params.Nysub), np.linspace(params.z0sub,params.z1sub,params.Nzsub))
        pos_sub = np.vstack([xi.ravel(),yi.ravel(),zi.ravel()]).T
        # detect gamma values
        
        #gamma = []
        gamma_x = []
        gamma_y = []
        gamma_z = []
        for p in tqdm(pos_sub,desc=' Vortex detection',leave=True,position=0):
            #IDs = np.argwhere( (pos[:,0]>=p[0]-params.R/2-1e-6) & (pos[:,1]>=p[1]-params.R/2-1e-6) & (pos[:,2]>=p[2]-params.R/2-1e-6) & 
            #                   (pos[:,0]<=p[0]+params.R/2+1e-6) & (pos[:,1]<=p[1]+params.R/2+1e-6) & (pos[:,2]<=p[2]+params.R/2+1e-6) )[:,0]
            #R, U = pos[IDs]-p, vel[IDs]
            #cross = np.cross(R,U,axis=1)
            #Rmag, Umag = np.linalg.norm(R,axis=1) , np.linalg.norm(U,axis=1)
            #Rmag[Rmag==0] = 1e-8
            #Umag[Umag==0] = 1e-8
            #phi = (cross[:,0]/np.sqrt(3) + cross[:,1]/np.sqrt(3) + cross[:,2]/np.sqrt(3)) / (Rmag*Umag)
            #gamma.append(np.nanmean(phi))
            # slice x
            ID = np.argwhere(np.abs(x-p[0])<5)[:,0]
            xi, yi, vxi, vyi = y[ID], z[ID], vy[ID], vz[ID]
            IDs = np.argwhere( (xi>=p[1]-params.R/2-1e-6) & (yi>=p[2]-params.R/2-1e-6) & (xi<=p[1]+params.R/2+1e-6) & (yi<=p[2]+params.R/2+1e-6) )[:,0]
            Rj, uj = np.vstack([xi[IDs],yi[IDs]]).T-np.array([p[1],p[2]]),  np.vstack([vxi[IDs],vyi[IDs]]).T
            phij_x = np.cross(Rj,uj) / np.linalg.norm(Rj,axis=1) / np.linalg.norm(uj,axis=1)
            gamma_x.append(np.nanmean(phij_x))
            # slice y
            ID = np.argwhere(np.abs(y-p[1])<5)[:,0]
            xi, yi, vxi, vyi = x[ID], z[ID], vx[ID], vz[ID]
            IDs = np.argwhere( (xi>=p[0]-params.R/2-1e-6) & (yi>=p[2]-params.R/2-1e-6) & (xi<=p[0]+params.R/2+1e-6) & (yi<=p[2]+params.R/2+1e-6) )[:,0]
            Rj, uj = np.vstack([xi[IDs],yi[IDs]]).T-np.array([p[0],p[2]]),  np.vstack([vxi[IDs],vyi[IDs]]).T
            phij_y = np.cross(Rj,uj) / np.linalg.norm(Rj,axis=1) / np.linalg.norm(uj,axis=1)
            gamma_y.append(np.nanmean(phij_y))
            # slice z
            ID = np.argwhere(np.abs(z-p[2])<5)[:,0]
            xi, yi, vxi, vyi = x[ID], y[ID], vx[ID], vy[ID]
            IDs = np.argwhere( (xi>=p[0]-params.R/2-1e-6) & (yi>=p[1]-params.R/2-1e-6) & (xi<=p[0]+params.R/2+1e-6) & (yi<=p[1]+params.R/2+1e-6) )[:,0]
            Rj, uj = np.vstack([xi[IDs],yi[IDs]]).T-np.array([p[0],p[1]]),  np.vstack([vxi[IDs],vyi[IDs]]).T
            phij_z = np.cross(Rj,uj) / np.linalg.norm(Rj,axis=1) / np.linalg.norm(uj,axis=1)
            gamma_z.append(np.nanmean(phij_z))
        #gamma = np.asarray(gamma)
        gammax = np.asarray(gamma_x)
        gammay = np.asarray(gamma_y)
        gammaz = np.asarray(gamma_z)
        gamma = np.sqrt(gammax**2+gammay**2+gammaz**2)
        np.savetxt(params.field_path+"VortexDetection/gamma_"+str(t)+'.txt', gamma)
        # append center of vortex
        center.append( pos_sub[np.argmax(np.abs(gamma))] )
    center = np.asarray(center)
    
    # plot
    ax = plt.figure(figsize=(12,12)).add_subplot(111, projection='3d')
    [ax.scatter(p[0],p[1],p[2], c='red') for p in center]
    ax.plot(center[:,0],center[:,1],center[:,2], c='black')
    ax.set_xlim(params.x0,params.x1), ax.set_ylim(params.y0,params.y1), ax.set_zlim(params.z0,params.z1)
    plt.tight_layout(), plt.show()
    
    # save out
    np.savetxt(params.field_path+"VortexDetection/center_R"+str(params.R)+'.txt', center)
if __name__ == "__main__":
    main()