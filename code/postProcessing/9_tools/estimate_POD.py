'''

    This Code estimates a POD of the divfree Euler fields.
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

os.chdir('../../../data')


# %%

class POD_parameter():    
    case_name, runname = 'rbc_300mm_run2', 'proPTV_reversal_20000_30000'
    t_start, t_end, dt = 20010, 20015, 1
    
    x0, x1, Nx = 0, 300, 32
    y0, y1, Ny = 0, 300, 32
    z0, z1, Nz = 0, 300, 32
    
    mode = 0
    
    plotmode = 'offdiag' # 'diag' , 'offdiag' , 'slice'
    Nx0, Nx1 = 15, 16
    Ny0, Ny1 = 0, None
    Nz0, Nz1 = 0, None
    
    xplot, yplot, cplot = 1, 2, 2 # [0,1,2], [0,1,2] , [0,1,2,3]
    cmin, cmax, cnorm, colormap = -1.5, 1.5, 0, 'seismic'

# %%


def main(): 
    # load parameter
    params = POD_parameter()    
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/Euler_divfree/Euler_t_{time}_divfree.txt"
    
    # make POD matrix M
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    M = np.zeros([3*params.Nx*params.Ny*params.Nz,len(times)])
    print('POD:')
    
    if os.path.isfile(params.case_name+'/analysis/POD.npz'):
        XYZ = np.loadtxt(params.field_path.format(time=str(int(params.t_start))))[:,0:3]
        M = np.load(params.case_name+'/analysis/POD.npz')['data']
        print('loaded POD')
    else:  
        for i,t in enumerate(tqdm(times, desc='Load', position=0, leave=True)):
            #print(' t = ' + str(t))
            # load velocity field
            data = np.loadtxt(params.field_path.format(time=str(int(t))))
            XYZ = data[:,0:3]
            M[:,i] = data[:,3:6].ravel()   
        np.savez(params.case_name+'/analysis/POD.npz',data=M)
    
    # POD
    #M = (M - M.mean(axis = 0))
    U ,  eigenVal , eigenVec = np.linalg.svd(M, full_matrices = False)	
    # fft aus eigenVec
    S = np.matmul(np.diag(eigenVal),eigenVec)
    V = np.matmul(U,np.diag(eigenVal))[:,params.mode].reshape(len(XYZ),3)
    modeGrid = np.append(XYZ,V,axis=1)
        
    # plot 3d 
    N = 4
    for m in range(4):
        V_m = np.matmul(U,np.diag(eigenVal))[:,m].reshape(len(XYZ),3)
        c = V_m[:,2]
        c = plt.cm.seismic(c)
        V_mag = 1#np.linalg.norm(V_m,axis=1)
        ax = plt.figure(figsize=(12,12)).add_subplot(111, projection='3d')
        #ax.quiver(modeGrid[::N,0], modeGrid[::N,1], modeGrid[::N,2], V_m[::N,0]/V_mag, V_m[::N,1]/V_mag, V_m[::N,2]/V_mag, color=c,norm=TwoSlopeNorm(0))
        ax.scatter(modeGrid[::N,0], modeGrid[::N,1], modeGrid[::N,2], c=V_m[::N,2],cmap='seismic',norm=TwoSlopeNorm(0))
        #try scatter
        ax.set_xlim(params.x0,params.x1), ax.set_ylim(params.y0,params.y1), ax.set_zlim(params.z0,params.z1)
        plt.tight_layout(), plt.show()
    
    # plot energy of each mode
    plt.figure()
    plt.bar(np.linspace(0,len(eigenVal)-1,len(eigenVal)),eigenVal)
    plt.show()
    
    # plot selected mode
    pos, vel = modeGrid[:,:3:], modeGrid[:,3:6:]
    vel /= np.max(vel)
    if params.plotmode == 'slice':
        X, Y = pos[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], pos[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        U, V = vel[:,params.xplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1], vel[:,params.yplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        C = vel[:,params.cplot].reshape(params.Nx,params.Ny,params.Nz)[params.Nx0:params.Nx1,params.Ny0:params.Ny1,params.Nz0:params.Nz1]
        Levels = np.linspace(params.cmin,params.cmax,2001)
        Norm = TwoSlopeNorm(params.cnorm)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=Levels, cmap=params.colormap, norm=Norm)
        plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
        cbar = plt.colorbar(contour)
        cbar.set_label(['vx','vy','vz'][params.cplot])
        plt.tight_layout(), plt.show()
    elif params.plotmode == 'diag':
        ID = np.argwhere(np.abs(pos[:,0]-pos[:,1])==0)[:,0]
        X, Y = np.sqrt(pos[ID,0]**2+pos[ID,1]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
        U, V = ((vel[ID,0]+vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
        C = vel[ID,params.cplot].reshape(params.Nx,params.Nz)
        Levels = np.linspace(params.cmin,params.cmax,2001)
        Norm = TwoSlopeNorm(params.cnorm)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=Levels, cmap=params.colormap, norm=Norm)
        plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
        cbar = plt.colorbar(contour)
        cbar.set_label(['vx','vy','vz'][params.cplot])
        plt.tight_layout(), plt.show()
    elif params.plotmode == 'offdiag':
        ID = np.argwhere(np.abs(pos[:,1]-(params.x1-pos[:,0]))<1e-8)[:,0] 
        X, Y = np.sqrt(pos[ID,0]**2+pos[ID,0]**2).reshape(params.Nx,params.Nz), pos[ID,2].reshape(params.Nx,params.Nz)
        U, V = ((vel[ID,0]-vel[ID,1])/np.sqrt(2)).reshape(params.Nx,params.Nz), vel[ID,2].reshape(params.Nx,params.Nz)
        C = vel[ID,params.cplot].reshape(params.Nx,params.Nz)
        Levels = np.linspace(params.cmin,params.cmax,2001)
        Norm = TwoSlopeNorm(params.cnorm)
        # plot
        plt.figure(figsize=(10,8))
        contour = plt.contourf(X.squeeze(), Y.squeeze(), C.squeeze(), levels=Levels, cmap=params.colormap, norm=Norm)
        plt.quiver(X.squeeze(), Y.squeeze(), U.squeeze(), V.squeeze(), alpha=1, color ='black', zorder=100)    
        cbar = plt.colorbar(contour)
        cbar.set_label(['vx','vy','vz'][params.cplot])
        plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()