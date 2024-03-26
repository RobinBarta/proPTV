'''

    This Code estimates a temperature field based on the Euler fields.
    
'''


import os
import numpy as np

os.chdir('../5_assimilatePressure')
from pressure_functions import EstimateTerms

os.chdir('../../../data')


# %%

class Temperature_parameter():    
    case_name, runname = 'syn_8000_20', 'proPTV_8000_0_10'
    t_start, t_end, dt = 0, 10, 1
    
    Pr , Ra, timestep = 6.9 , 1E10, 0.06
    T0, T1 = -0.5, 0.5
    x0, x1, Nx = 0, 1, 32
    y0, y1, Ny = 0, 1, 32
    z0, z1, Nz = 0, 1, 32
    dx , dy , dz = (x1-x0)/(Nx-1) , (y1-y0)/(Ny-1) , (z1-z0)/(Nz-1)

# %%


def main(): 
    # load parameter
    params = Temperature_parameter()    
    params.field_path = params.case_name+"/output/"+params.runname+"/fields/"
    
    # create output folders
    os.makedirs(params.field_path+"Euler_temperature",exist_ok=True)
    
    # assimilate pressure field for each time step:
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    print('Estimate Temperature Field:')
    for i,t in enumerate(times):
        print(' t = ' + str(t))
        
        # load velocity field at time step i
        data = np.loadtxt(params.field_path+"Euler_divfree/"+"Euler_t_"+str(int(times[i]))+"_divfree.txt")
        x, y, z = data[:,0].reshape([params.Ny,params.Nx,params.Nz]), data[:,1].reshape([params.Nz,params.Ny,params.Nx]), data[:,2].reshape([params.Nz,params.Ny,params.Nx])
        vx0, vy0, vz0 = data[:,3].reshape([params.Ny,params.Nx,params.Nz]), data[:,4].reshape([params.Nz,params.Ny,params.Nx]), data[:,5].reshape([params.Nz,params.Ny,params.Nx])
        # load pressure field at time step i
        data = np.loadtxt(params.field_path+"Euler_divfree/"+"Euler_t_"+str(int(times[i+1]))+"_divfree.txt")
        p = data[:,3].reshape([params.Ny,params.Nx,params.Nz])
        # load velocity field at time step i+1
        data = np.loadtxt(params.field_path+"Euler_divfree/"+"Euler_t_"+str(int(times[i+1]))+"_divfree.txt")
        vx1, vy1, vz1 = data[:,3].reshape([params.Ny,params.Nx,params.Nz]), data[:,4].reshape([params.Nz,params.Ny,params.Nx]), data[:,5].reshape([params.Nz,params.Ny,params.Nx])
        
        # estimate gradients
        u , v , w , P = np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2])
        u[1:-1,1:-1,1:-1] , v[1:-1,1:-1,1:-1] , w[1:-1,1:-1,1:-1], P[1:-1,1:-1,1:-1] = vx0, vy0, vz0, p
        P[:,:,0], P[:,:,-1] = P[:,:,1], P[:,:,-2]
        P[:,0,:], P[:,-1,:] = P[:,1,:], P[:,-2,:]
        P[0,:,:], P[-1,:,:] = P[1,:,:], P[-2,:,:]
        u_conv, v_conv, w_conv, u_diff, v_diff, w_diff = EstimateTerms(u, v, w, params)
        dpdz = (P[1:-1,1:-1,2:]-P[1:-1,1:-1,1:-1])/params.dz
        dwdt = (-w_conv + w_diff)[1:-1,1:-1,1:-1] # (vz1-vz0)/params.timestep
        
        # estimate temperature
        T = dwdt + w_conv[1:-1,1:-1,1:-1] - w_diff[1:-1,1:-1,1:-1] + dpdz
        T[:,:,-1], T[:,:,0] = params.T0, params.T1
        
        # save output
        np.savetxt(params.field_path+"Euler_temperature/"+"Euler_t_"+str(int(t))+"_temperature.txt",np.vstack([np.ravel(x),np.ravel(y),np.ravel(z),np.ravel(T)]).T,header='x,y,z,T')    
        
        if i == len(times)-2:
            break
if __name__ == "__main__":
    main()