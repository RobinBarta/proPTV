'''

    This Code plots the LSC center.
    
'''


import os, sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

os.chdir('../../../data')


# %%

class LSC_parameter():    
    case_name, runname = 'rbc_300mm_run2', 'proPTV_reversal_40001_50000'
    t_start = [46000]
    
    x0, x1, Nx = 0, 300, 32
    y0, y1, Ny = 0, 300, 32
    z0, z1, Nz = 0, 300, 32
    
    plot = 0
    plotmode = 'diag' # 'diag' , 'offdiag' , 'slice'
    Nx0, Nx1 = 15, 16
    Ny0, Ny1 = 0, None
    Nz0, Nz1 = 0, None
    xplot, yplot, cplot = 1, 2, 2 # [0,1,2], [0,1,2] , [0-u,1-v,2-w,3-p]
    cmin, cmax, cnorm, colormap = -1.5, 1.5, 0, 'seismic'

# %%


def main(): 
    # load parameter
    params = LSC_parameter()    
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/POD/"
    
    for t in params.t_start:
        data = np.append(np.loadtxt(params.field_path+"POD_t_{time}.txt".format(time=str(int(t)))),np.loadtxt(params.field_path+"POD_t_{time}_pressure.txt".format(time=str(int(t)))).reshape(-1,1),axis=1)
        centers = np.loadtxt(params.field_path+"POD_t_{time}_centers.txt".format(time=str(int(t))))
        
        if params.plot == 0:
            # plot centers
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(centers[:,0], centers[:,1], centers[:,2], color='black', s=20)
            ax.set_xlim(params.x0,params.x1), ax.set_ylim(params.y0,params.y1), ax.set_zlim(params.z0,params.z1)
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            plt.tight_layout(), plt.show()
        
        elif params.plot == 1:
            # plot 3D
            ID = np.argwhere(data[:,0]>-1)[:,0]
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111, projection='3d')
            color = ax.scatter(data[ID,0],data[ID,1],data[ID,2],c=data[ID,3+params.cplot],cmap=params.colormap,norm=TwoSlopeNorm(params.cnorm))
            fig.colorbar(color, ax=ax)
            ax.set_xlim(params.x0,params.x1), ax.set_ylim(params.y0,params.y1), ax.set_zlim(params.z0,params.z1)
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            plt.tight_layout(), plt.show()
        
        elif params.plot == 2:
            # plot 2D 
            pos, vel = data[:,:3:], data[:,3::]
            vel /= np.max(vel)
            if params.plotmode == 'slice':
                X, Y = pos[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], pos[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
                U, V = vel[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], vel[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
                C = vel[:,params.cplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
                Levels = np.linspace(params.cmin,params.cmax,201)
                Norm = TwoSlopeNorm(params.cnorm)
                # plot
                plt.figure(figsize=(10,8))
                contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=Levels, cmap=params.colormap, norm=Norm)
                plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
                cbar = plt.colorbar(contour)
                cbar.set_label(['vx','vy','vz','p'][params.cplot])
                plt.tight_layout(), plt.show()
            elif params.plotmode == 'diag':
                ID = np.argwhere(np.abs(pos[:,0]-pos[:,1])==0)[:,0]
                X, Y = np.sqrt(pos[ID,0]**2+pos[ID,1]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
                U, V = ((vel[ID,0]+vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
                C = vel[ID,params.cplot].reshape(params.Nx,params.Nz)
                Levels = np.linspace(params.cmin,params.cmax,201)
                Norm = TwoSlopeNorm(params.cnorm)
                # plot
                plt.figure(figsize=(10,8))
                contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=Levels, cmap=params.colormap, norm=Norm)
                plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
                cbar = plt.colorbar(contour)
                cbar.set_label(['vx','vy','vz','p'][params.cplot])
                plt.tight_layout(), plt.show()
            elif params.plotmode == 'offdiag':
                ID = np.argwhere(np.abs(pos[:,1]-(params.x1-pos[:,0]))<1e-8)[:,0] 
                X, Y = np.sqrt(pos[ID,0]**2+pos[ID,0]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
                U, V = ((vel[ID,0]-vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
                C = vel[ID,params.cplot].reshape(params.Nx,params.Nz)
                Levels = np.linspace(params.cmin,params.cmax,201)
                Norm = TwoSlopeNorm(params.cnorm)
                # plot
                plt.figure(figsize=(10,8))
                contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=Levels, cmap=params.colormap, norm=Norm)
                plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
                cbar = plt.colorbar(contour)
                cbar.set_label(['vx','vy','vz','p'][params.cplot])
                plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()