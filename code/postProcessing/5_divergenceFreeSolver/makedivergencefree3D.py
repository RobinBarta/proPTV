'''

    This Code solves the pressure poisson equation on an euler grid to make field divergence free.
    
'''


import os
import numpy as np

from tqdm import tqdm

from makedivergencefree3D_functions import *

os.chdir('../../../data')


# %%

class Divfree_parameter():    
    case_name, runname = '27000', 'run1'
    t_start, t_end, dt = 10, 10, 1
    
    x0, x1, Nx = 0, 1, 50
    y0, y1, Ny = 0, 1, 50
    z0, z1, Nz = 0, 1, 50
    dx , dy , dz = x1/(Nx-1) , y1/(Ny-1) , z1/(Nz-1)

# %%


def main():
    # load parameter
    params = Divfree_parameter()
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/"
    
    # create output folders
    os.makedirs(params.field_path+"Euler_divfree",exist_ok=True)
    
    # correct velocity field for each time step:
    print('Div-Free Solver:')
    for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]:
        print(' t = ' + str(t))    
        # load Euler field
        x, y, z, vx, vy, vz = LoadEuler3D(t,params)
        # initialise variabels , including ghost cells
        u, v, w, p, u_sol, v_sol, w_sol = Initialise3DVariables(vx,vy,vz,params)
        # calculate divergence
        div = Divergence3D(u,v,w,params)
        # set up sparse matrix for direct poisson solver
        A = Get3DSparseMatrix(params)
        p_solve , info = scipy.sparse.linalg.bicg(A,np.ravel(div),tol=1e-10)
        p[1:-1,1:-1,1:-1] = p_solve.reshape([params.Nz,params.Ny,params.Nx])
        # correct velocity and estimate new divergence
        u_sol, v_sol, w_sol = HelmholtzHodgeCorrection(u_sol,v_sol,w_sol,u,v,w,p,params)
        div_sol = Divergence3D(u_sol,v_sol,w_sol,params)
        # check solution
        print('  div for: ' , np.mean(np.abs(div)), ' div after: ', np.mean(np.abs(div_sol)))
        # save output
        np.savetxt(params.field_path+"Euler_divfree/"+"Euler_t_"+str(int(t))+"_divfree.txt",np.vstack([np.ravel(x),np.ravel(y),np.ravel(z),np.ravel(u_sol[1:-1,1:-1,1:-1]),np.ravel(v_sol[1:-1,1:-1,1:-1]),np.ravel(w_sol[1:-1,1:-1,1:-1]),np.ravel(p[1:-1,1:-1,1:-1])]).T,header='x, y, z, vx, vy, vz, p')
if __name__ == "__main__":
    main()