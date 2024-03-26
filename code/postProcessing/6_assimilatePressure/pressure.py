'''

    This Code estimates a pressure field based on the Euler fields.
    
'''


import os, scipy.sparse.linalg
import numpy as np

from tqdm import tqdm

from pressure_functions import *
os.chdir('../5_divergenceFreeSolver')
from makedivergencefree3D_functions import Get3DSparseMatrix, Divergence3D

os.chdir('../../../data')


# %%

class Pressure_parameter():    
    case_name, runname = '27000', 'run1'
    t_start, t_end, dt = 10, 10, 1
    
    Pr , Ra, timestep = 6.9 , 1E10, 0.075
    x0, x1, Nx = 0, 1, 50
    y0, y1, Ny = 0, 1, 50
    z0, z1, Nz = 0, 1, 50
    dx , dy , dz = (x1-x0)/(Nx-1) , (y1-y0)/(Ny-1) , (z1-z0)/(Nz-1)

# %%


def main(): 
    # load parameter
    params = Pressure_parameter()    
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/"
    
    # create output folders
    os.makedirs(params.field_path+"Euler_pressure",exist_ok=True)
    
    # assimilate pressure field for each time step:
    print('Estimate Pressure Field:')
    for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]:
        print(' t = ' + str(t))
        # load velocity field
        data = np.loadtxt(params.field_path+"Euler_divfree/"+"Euler_t_"+str(int(t))+"_divfree.txt")
        x, y, z = data[:,0].reshape([params.Ny,params.Nx,params.Nz]), data[:,1].reshape([params.Nz,params.Ny,params.Nx]), data[:,2].reshape([params.Nz,params.Ny,params.Nx])
        vx, vy, vz = data[:,3].reshape([params.Ny,params.Nx,params.Nz]), data[:,4].reshape([params.Nz,params.Ny,params.Nx]), data[:,5].reshape([params.Nz,params.Ny,params.Nx])
        u , v , w , p = np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz,params.Ny,params.Nx])
        u[1:-1,1:-1,1:-1] , v[1:-1,1:-1,1:-1] , w[1:-1,1:-1,1:-1] = vx, vy, vz
        
        # calculate fractional step
        u_conv, v_conv, w_conv, u_diff, v_diff, w_diff = EstimateTerms(u, v, w, params)
        ut = u.copy() + params.timestep * (-u_conv + u_diff)
        vt = v.copy() + params.timestep * (-v_conv + v_diff)
        wt = w.copy() + params.timestep * (-w_conv + w_diff)
        div = Divergence3D(ut,vt,wt,params) / params.timestep
        
        # set up sparse matrix and divergence for direct poisson solver
        A = Get3DSparseMatrix(params)
        p_solve , info = scipy.sparse.linalg.bicg(A,np.ravel(div),tol=1e-10)
        p = p_solve.reshape([params.Nx,params.Ny,params.Nz])
        # save output
        np.savetxt(params.field_path+"Euler_pressure/"+"Euler_t_"+str(int(t))+"_pressure.txt",np.vstack([np.ravel(x),np.ravel(y),np.ravel(z),np.ravel(p)]).T,header='x,y,z,p')    
if __name__ == "__main__":
    main()