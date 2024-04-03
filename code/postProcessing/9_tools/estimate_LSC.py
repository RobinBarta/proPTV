'''

    This Code estimates centers of LSC.
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.ndimage import minimum_filter
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

os.chdir('../5_divergenceFreeSolver')
from makedivergencefree3D_functions import Get3DSparseMatrix, Divergence3D
os.chdir('../6_assimilatePressure')
from pressure_functions import *
os.chdir('../../../data')


# %%

class LSC_parameter():    
    case_name, runname = 'rbc_300mm_run2', 'proPTV_reversal_40001_50000'
    t_start, t_end, dt = 46000, 47000, 50
    
    Pr , Ra, timestep = 6.9 , 1E9, 0.2
    x0, x1, Nx = 0, 300, 32
    y0, y1, Ny = 0, 300, 32
    z0, z1, Nz = 0, 300, 32
    dx , dy , dz = (x1-x0)/(Nx-1) , (y1-y0)/(Ny-1) , (z1-z0)/(Nz-1)
    
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
    params = LSC_parameter()    
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/"
    
    # create output folders
    os.makedirs(params.field_path+"POD",exist_ok=True)
    
    print('POD:')
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    M = np.zeros([3*params.Nx*params.Ny*params.Nz,len(times)])
    for i,t in enumerate(tqdm(times, desc='Load', position=0, leave=True)):
        data = np.loadtxt(params.field_path+"Euler_divfree/Euler_t_{time}_divfree.txt".format(time=str(int(t))))
        XYZ = data[:,0:3]
        M[:,i] = data[:,3:6].ravel()  
    #M = (M - M.mean(axis = 0))
    U ,  eigenVal , eigenVec = np.linalg.svd(M, full_matrices = False)	
    S = np.matmul(np.diag(eigenVal),eigenVec)
    V = np.matmul(U,np.diag(eigenVal))[:,params.mode].reshape(len(XYZ),3)
    modes = np.append(XYZ,V,axis=1)
    np.savetxt(params.field_path+"POD/POD_t_{time}.txt".format(time=str(int(params.t_start))),modes)
    
    print('Pressure estimation:')
    x, y, z = modes[:,0].reshape([params.Ny,params.Nx,params.Nz]), modes[:,1].reshape([params.Nz,params.Ny,params.Nx]), modes[:,2].reshape([params.Nz,params.Ny,params.Nx])
    vx, vy, vz = modes[:,3].reshape([params.Ny,params.Nx,params.Nz]), modes[:,4].reshape([params.Nz,params.Ny,params.Nx]), modes[:,5].reshape([params.Nz,params.Ny,params.Nx])
    u , v , w , p = np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz,params.Ny,params.Nx])
    u[1:-1,1:-1,1:-1] , v[1:-1,1:-1,1:-1] , w[1:-1,1:-1,1:-1] = vx, vy, vz
    u_conv, v_conv, w_conv, u_diff, v_diff, w_diff = EstimateTerms(u, v, w, params)
    ut = u.copy() + params.timestep * (-u_conv + u_diff)
    vt = v.copy() + params.timestep * (-v_conv + v_diff)
    wt = w.copy() + params.timestep * (-w_conv + w_diff)
    div = Divergence3D(ut,vt,wt,params) / params.timestep
    A = Get3DSparseMatrix(params)
    p_solve , info = scipy.sparse.linalg.bicg(A,np.ravel(div),tol=1e-10)
    p = p_solve.reshape([params.Nx,params.Ny,params.Nz])
    p = p-(np.max(p)+np.min(p))/2
    p = p/np.max(p)
    np.savetxt(params.field_path+"POD/POD_t_{time}_pressure.txt".format(time=str(int(params.t_start))),p.ravel())
    print(' - done')

    print('Find vortex centers:')
    local_minima_mask = (p == minimum_filter(p, size=(3,3,3)))
    minima_locations = np.where(local_minima_mask)
    centers = np.vstack([x[minima_locations], y[minima_locations], z[minima_locations]]).T
    np.savetxt(params.field_path+"POD/POD_t_{time}_centers.txt".format(time=str(int(params.t_start))),centers)
    print(' - done')
if __name__ == "__main__":
    main()