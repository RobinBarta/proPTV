'''

    This Code creates a data set which can be loaded by our PINN to assimilate T.
    
'''


import os
import numpy as np

from tqdm import tqdm

os.chdir('../../../data')


# %%

class Data_parameter():    
    case_name, runname = 'syn_8000_20', 'proPTV_8000_0_10'
    t_start, t_end, dt = 0, 9, 1
    
    Nx, Ny, Nz = 32, 32, 32

# %%


def main():
    # load parameter
    params = Data_parameter()
    
    # make output data
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    PINNdata = np.zeros([params.Nx,params.Ny,params.Nz,len(times),9])
    
    # fill data
    for i,t in enumerate(tqdm(times, desc='Create dataset for PINN', position=0 , leave=True, delay=0.5)):
        # load
        data = np.loadtxt( params.case_name+"/output/"+params.runname+"/fields/Euler_divfree/Euler_t_{time}_divfree.txt".format(time=int(t)))
        x, y, z, u, v, w = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        data = np.loadtxt( params.case_name+"/output/"+params.runname+"/fields/Euler_pressure/Euler_t_{time}_pressure.txt".format(time=int(t)))
        p = data[:,3]
        data = np.loadtxt( params.case_name+"/output/"+params.runname+"/fields/Euler_temperature/Euler_t_{time}_temperature.txt".format(time=int(t)))
        T = data[:,3]
        # fill
        PINNdata[:,:,:,i,0] = t
        PINNdata[:,:,:,i,1], PINNdata[:,:,:,i,2], PINNdata[:,:,:,i,3] = x.reshape(params.Nx,params.Ny,params.Nz), y.reshape(params.Nx,params.Ny,params.Nz), z.reshape(params.Nx,params.Ny,params.Nz)
        PINNdata[:,:,:,i,4], PINNdata[:,:,:,i,5], PINNdata[:,:,:,i,6] = u.reshape(params.Nx,params.Ny,params.Nz), v.reshape(params.Nx,params.Ny,params.Nz), w.reshape(params.Nx,params.Ny,params.Nz)
        PINNdata[:,:,:,i,7], PINNdata[:,:,:,i,8] = T.reshape(params.Nx,params.Ny,params.Nz), p.reshape(params.Nx,params.Ny,params.Nz)
    
    # save data
    np.savez(params.case_name+'/analysis/PINN_dataset_N_'+str(params.Nx)+'_t0_'+str(params.t_start)+'_t1_'+str(params.t_end)+'.npz',data=PINNdata)
if __name__ == "__main__":
    main()