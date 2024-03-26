'''

    This Code visualizes the pressure.
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

os.chdir('../../../data')


# %%

class Interpolation_parameter():
    case_name, runname = '27000', 'run1'
    
    t = 10
    x0, x1, Nx = 0, 1, 50
    y0, y1, Ny = 0, 1, 50
    z0, z1, Nz = 0, 1, 50
    
    plotmode = 'offdiag' # 'diag' , 'offdiag' , 'slice'
    Nx0, Nx1 = 15, 16
    Ny0, Ny1 = 0, None
    Nz0, Nz1 = 0, None
    
    xplot, yplot = 1, 2 # [0,1,2], [0,1,2]
    colormap ='jet'

# %%


def main():
    # load parameter
    params = Interpolation_parameter()
    # load data
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/Euler_pressure/Euler_t_{time}_pressure.txt"
    data = np.loadtxt(params.field_path.format(time=int(params.t)))
    x, y, z, p = data[:,0], data[:,1], data[:,2], data[:,3]
    pos = np.vstack([x,y,z]).T
    data = np.loadtxt(params.case_name+"/output/"+params.runname+"/fields/Euler_divfree/Euler_t_{time}_divfree.txt".format(time=int(params.t)))
    vx, vy, vz = data[:,3], data[:,4], data[:,5]
    vel = np.vstack([vx,vy,vz,p]).T
    
    # plot
    if params.plotmode == 'slice':
        X, Y = pos[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], pos[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        U, V = vel[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], vel[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        P = p.reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), P.squeeze(), params.Nx*params.Ny, cmap=params.colormap)
        plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
        cbar = plt.colorbar(contour)
        cbar.set_label('p')
        plt.tight_layout(), plt.show()
    elif params.plotmode == 'diag':
        ID = np.argwhere(np.abs(pos[:,0]-pos[:,1])==0)[:,0]
        X, Y = np.sqrt(pos[ID,0]**2+pos[ID,1]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
        U, V = ((vel[ID,0]+vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
        P = p[ID].reshape(params.Nx,params.Nz)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), P.squeeze(), params.Nx*params.Ny, cmap=params.colormap)
        plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
        cbar = plt.colorbar(contour)
        cbar.set_label('p')
        plt.tight_layout(), plt.show()
    elif params.plotmode == 'offdiag':
        ID = np.argwhere(np.abs(pos[:,1]-(params.x1-pos[:,0]))<1e-8)[:,0] 
        X, Y = np.sqrt(pos[ID,0]**2+pos[ID,0]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
        U, V = ((vel[ID,0]-vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
        C = p[ID].reshape(params.Nx,params.Nz)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), params.Nx*params.Ny, cmap=params.colormap)
        plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
        cbar = plt.colorbar(contour)
        cbar.set_label('p')
        plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()