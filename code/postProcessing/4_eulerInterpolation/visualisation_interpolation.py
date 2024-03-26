'''

    This Code visualizes the interpolation.
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

os.chdir('../../../data')


# %%

class Interpolation_parameter():
    case_name, runname = '27000', 'run1'
    mode = 'euler' # 'lagrange' ,  'euler'
    
    t = 10
    x0, x1, Nx = 0, 1, 50
    y0, y1, Ny = 0, 1, 50
    z0, z1, Nz = 0, 1, 50
    
    plotmode = 'offdiag' # 'diag' , 'offdiag' , 'slice'
    Nx0, Nx1 = 15, 16
    Ny0, Ny1 = 0, None
    Nz0, Nz1 = 0, None
    
    xplot, yplot, cplot = 1, 2, 2 # [0,1,2], [0,1,2] , [0,1,2]
    cmin, cmax, cnorm, colormap = -0.015, 0.015, 0, 'seismic'

# %%


def main():
    # load parameter
    params = Interpolation_parameter()
    
    if params.mode == 'euler':
        # load data
        params.field_path = params.case_name+"/output/"+params.runname+"/fields/Euler/Euler_t_{time}.txt"
        data = np.loadtxt(params.field_path.format(time=int(params.t)),skiprows=1)
        x, y, z, vx, vy, vz = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        pos, vel = np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T
        # plot
        if params.plotmode == 'slice':
            X, Y = pos[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], pos[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
            U, V = vel[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], vel[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
            C = vel[:,params.cplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
            # plot
            plt.figure(figsize=(10,8))
            contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=np.linspace(params.cmin,params.cmax,2001), cmap=params.colormap, norm=TwoSlopeNorm(params.cnorm))
            plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
            cbar = plt.colorbar(contour)
            cbar.set_label(['vx','vy','vz'][params.cplot])
            plt.tight_layout(), plt.show()
        elif params.plotmode == 'diag':
            ID = np.argwhere(np.abs(pos[:,0]-pos[:,1])==0)[:,0]
            X, Y = (1/np.sqrt(2))*np.sqrt(pos[ID,0]**2+pos[ID,1]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
            U, V = ((vel[ID,0]+vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
            #np.savetxt(params.case_name+"/output/"+params.runname+"/fields/Euler/dns_diag.txt",np.vstack([X.ravel(),Y.ravel(),U.ravel(),V.ravel()]).T)
            # plot
            plt.figure(figsize=(10,8))
            contour = plt.contourf(X.squeeze(), Y.squeeze(), V.squeeze(), levels=np.linspace(params.cmin,params.cmax,2001), cmap=params.colormap, norm=TwoSlopeNorm(params.cnorm))
            plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
            cbar = plt.colorbar(contour)
            cbar.set_label('vz')
            plt.tight_layout(), plt.show()
        elif params.plotmode == 'offdiag':
            ID = np.argwhere(np.abs(pos[:,1]-(params.x1-pos[:,0]))<1e-3)[:,0]
            X, Y = (1/np.sqrt(2))*np.sqrt(pos[ID,0]**2+pos[ID,0]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
            U, V = ((vel[ID,0]-vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
            #np.savetxt(params.case_name+"/output/"+params.runname+"/fields/Euler/dns_off.txt",np.vstack([X.ravel(),Y.ravel(),U.ravel(),V.ravel()]).T)
            # plot
            plt.figure(figsize=(10,8))
            contour = plt.contourf(X.squeeze(), Y.squeeze(), V.squeeze(), levels=np.linspace(params.cmin,params.cmax,2001), cmap=params.colormap, norm=TwoSlopeNorm(params.cnorm))
            plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
            cbar = plt.colorbar(contour)
            cbar.set_label('vz')
            plt.tight_layout(), plt.show()
    
    elif params.mode == 'lagrange':
        # load data
        params.field_path = params.case_name+"/output/"+params.runname+"/fields/Lagrange/Lagrange_t_{time}.txt"
        data = np.loadtxt(params.field_path.format(time=int(params.t)),skiprows=1)
        x, y, z, vx, vy, vz = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        vel = np.vstack([vx,vy,vz]).T
        # plot
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x,y,z, c=vel[:,params.cplot] ,cmap='seismic')
        ax.set_xlim(params.x0,params.x1), ax.set_ylim(params.y0,params.y1), ax.set_zlim(params.z0,params.z1)
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        plt.show()
if __name__ == "__main__":
    main()