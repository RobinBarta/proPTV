import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

def LoadNETCDF(t):
    path_data = '../input/PARTICLE_{time}.nc'
    data = nc.Dataset(path_data.format(time=str(int(t)).zfill(8)))
    Gr, Pr,  = data.variables['gr'][0], data.variables['pr'][0]
    ti, x, y, z = data.variables['tprobl'][0], data['particles'][:,2], data['particles'][:,1], data['particles'][:,0]
    u, v, w, T, p = data['particles'][:,5], data['particles'][:,4], data['particles'][:,3], data['particles'][:,6], data['particles'][:,7]
    print(ti)
    return x,y,z,u,v,w

def main(): 
    
    n = 2
    t0, t1, dt = 540000, 541500, 5e-5*60*20
    #x0,y0,z0,u0,v0,w0 = LoadParticles(t0)
    #x1,y1,z1,u1,v1,w1 = LoadParticles(t1)
    x0,y0,z0,u0,v0,w0 = LoadNETCDF(t0)
    x1,y1,z1,u1,v1,w1 = LoadNETCDF(t1)
    
    err= np.linalg.norm( np.vstack([x1,y1,z1]).T-(np.vstack([x0,y0,z0]).T+dt*np.vstack([u0,v0,w0]).T) ,axis=1)
    
    plt.figure()
    plt.plot()
    plt.hist(err,bins=100)
    plt.ylabel('counts')
    plt.semilogy()
    plt.xlabel(r'$|x_{i+1} - (x_{i} + u_{i}\cdot \Delta t)|$ [mm]')
    plt.show() 
if __name__ == "__main__":
    main()