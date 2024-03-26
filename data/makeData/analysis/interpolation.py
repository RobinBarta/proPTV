'''

    This Code interpolates origin to an Euler grid and an Lagrange Grid.
    
'''


import os
import numpy as np

from tqdm import tqdm

from interpolation_functions import *


# %%

class Interpolation_parameter():    
    case_name, Zeros = '27000_25', 5
    t_start, t_end, dt = 10, 10, 1
    
    interpolationMode, smooth = 'nd', 0.0005 # rbf or nd
    dN = 1
    x0, x1, Nx = 0, 1, 50
    y0, y1, Ny = 0, 1, 50
    z0, z1, Nz = 0, 1, 50

# %%


def main():
    # load parameter
    params = Interpolation_parameter()
    params.field_path = '../output/'+params.case_name+'/fields/'
    params.data_path = '../output/'+params.case_name+'/origin/origin_{time}.txt'
    
    # create output folders
    os.makedirs(params.field_path+"Euler",exist_ok=True), os.makedirs(params.field_path+"Lagrange",exist_ok=True)
    
    # interpolation
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    PINNdata = np.zeros([params.Nx,params.Ny,params.Nz,len(times),9])
    print('Interpolation:')
    for i,t in enumerate(times):
        print(' t = ' + str(t))
        # load data 
        data = np.loadtxt(params.data_path.format(time=str(t).zfill(params.Zeros)))[:,1:9:]
        x, y, z, u, v, w, T, p = InterpolateOriginToGrid(t,data,params) 
        # fill PINN data
        PINNdata[:,:,:,i,0] = t
        PINNdata[:,:,:,i,1], PINNdata[:,:,:,i,2], PINNdata[:,:,:,i,3] = x.reshape(params.Nx,params.Ny,params.Nz), y.reshape(params.Nx,params.Ny,params.Nz), z.reshape(params.Nx,params.Ny,params.Nz)
        PINNdata[:,:,:,i,4], PINNdata[:,:,:,i,5], PINNdata[:,:,:,i,6] = u.reshape(params.Nx,params.Ny,params.Nz), v.reshape(params.Nx,params.Ny,params.Nz), w.reshape(params.Nx,params.Ny,params.Nz)
        PINNdata[:,:,:,i,7], PINNdata[:,:,:,i,8] = T.reshape(params.Nx,params.Ny,params.Nz), p.reshape(params.Nx,params.Ny,params.Nz)
    np.savez('../output/'+params.case_name+'/PINN_dataset_N_'+str(params.Nx)+'_t0_'+str(params.t_start)+'_t1_'+str(params.t_end)+'.npz',data=PINNdata)
if __name__ == "__main__":
    main()