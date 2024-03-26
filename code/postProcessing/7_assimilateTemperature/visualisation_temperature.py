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
    case_name, runname = 'syn_8000_20', 'proPTV_8000_0_10'
    
    t = 0
    x0, x1, Nx = 0, 1, 32
    y0, y1, Ny = 0, 1, 32
    z0, z1, Nz = 0, 1, 32
    
    plotmode = 'slice' # 'diag' , 'offdiag' , 'slice'
    Nx0, Nx1 = 15, 16
    Ny0, Ny1 = 0, None
    Nz0, Nz1 = 0, None
    
    xplot, yplot = 1, 2 # [0,1,2], [0,1,2]
    colormap ='seismic'

# %%


def main():
    # load parameter
    params = Interpolation_parameter()
    # load data
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/Euler_temperature/Euler_t_{time}_temperature.txt"
    data = np.loadtxt(params.field_path.format(time=int(params.t)))
    x, y, z, T = data[:,0], data[:,1], data[:,2], data[:,3]
    pos = np.vstack([x,y,z]).T
    
    # plot
    if params.plotmode == 'slice':
        X, Y = pos[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], pos[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        T = T.reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), T.squeeze(), params.Nx*params.Ny, cmap=params.colormap)
        cbar = plt.colorbar(contour)
        cbar.set_label('T')
        plt.tight_layout(), plt.show()
    elif params.plotmode == 'diag':
        ID = np.argwhere(np.abs(pos[:,0]-pos[:,1])==0)[:,0]
        X, Y = np.sqrt(pos[ID,0]**2+pos[ID,1]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
        C = T[ID].reshape(params.Nx,params.Nz)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), params.Nx*params.Ny, cmap=params.colormap)
        cbar = plt.colorbar(contour)
        cbar.set_label('T')
        plt.tight_layout(), plt.show()
    elif params.plotmode == 'offdiag':
        ID = np.argwhere(np.abs(pos[:,1]-(1-pos[:,0]))<1e-8)[:,0] 
        X, Y = np.sqrt(pos[ID,0]**2+pos[ID,0]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
        C = T[ID].reshape(params.Nx,params.Nz)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), params.Nx*params.Ny, cmap=params.colormap)
        cbar = plt.colorbar(contour)
        cbar.set_label('T')
        plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()