'''

    This script contains interpolation functions.
    
'''


import numpy as np

from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


def InterpolateOriginToGrid(t,data,params):
    # save Lagrange grid
    np.savetxt(params.field_path+'Lagrange/Lagrange_t_'+str(int(t))+'.txt',data,header='x, y, z, vx, vy, vz, T, p')
    
    # load track information 
    X , Y , Z = data[::params.dN,0] , data[::params.dN,1] , data[::params.dN,2]
    VX , VY , VZ = data[::params.dN,3] , data[::params.dN,4] , data[::params.dN,5]
    T, P = data[::params.dN,6] , data[::params.dN,7]
        
    # fill in zero boundary condition
        # y-z BC
    y_bc, z_bc = np.meshgrid(np.linspace(params.y0,params.y1,params.Ny),np.linspace(params.z0,params.z1,params.Nz))
    VX0 , VY0 , VZ0, T0, P0 = np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.mean(P)*np.ones_like(y_bc)
    VX1 , VY1 , VZ1, T1, P1 = np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.mean(P)*np.ones_like(y_bc)
    X, Y, Z, VX, VY, VZ, T, P = np.append(X,np.ravel(params.x0*np.ones(params.Ny*params.Nz))), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX0)), np.append(VY,np.ravel(VY0)), np.append(VZ,np.ravel(VZ0)), np.append(T,np.ravel(T0)), np.append(P,np.ravel(P0))
    X, Y, Z, VX, VY, VZ, T, P = np.append(X,np.ravel(params.x1*np.ones(params.Ny*params.Nz))), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX1)), np.append(VY,np.ravel(VY1)), np.append(VZ,np.ravel(VZ1)), np.append(T,np.ravel(T1)), np.append(P,np.ravel(P1))
        # x-z BC    
    x_bc, z_bc = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.z0,params.z1,params.Nz))
    VX0 , VY0 , VZ0, T0, P0 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.mean(P)*np.ones_like(x_bc)   
    VX1 , VY1 , VZ1, T1, P1 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.mean(P)*np.ones_like(x_bc)   
    X, Y, Z, VX, VY, VZ, T, P = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(params.y0*np.ones(params.Nx*params.Nz))), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX0)), np.append(VY,np.ravel(VY0)), np.append(VZ,np.ravel(VZ0)), np.append(T,np.ravel(T0)), np.append(P,np.ravel(P0))
    X, Y, Z, VX, VY, VZ, T, P = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(params.y1*np.ones(params.Nx*params.Nz))), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX1)), np.append(VY,np.ravel(VY1)), np.append(VZ,np.ravel(VZ1)), np.append(T,np.ravel(T1)), np.append(P,np.ravel(P1)) 
        # x-y BC    
    x_bc, y_bc = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.y0,params.y1,params.Ny))
    VX0 , VY0 , VZ0, T0, P0 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) , -0.5*np.ones_like(x_bc) , np.mean(P)*np.ones_like(x_bc)  
    VX1 , VY1 , VZ1, T1, P1 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) , 0.5*np.ones_like(x_bc) , np.mean(P)*np.ones_like(x_bc)  
    X, Y, Z, VX, VY, VZ, T, P = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(params.z0*np.ones(params.Nx*params.Ny))), np.append(VX,np.ravel(VX0)), np.append(VY,np.ravel(VY0)), np.append(VZ,np.ravel(VZ0)), np.append(T,np.ravel(T0)), np.append(P,np.ravel(P0))
    X, Y, Z, VX, VY, VZ, T, P = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(params.z1*np.ones(params.Nx*params.Ny))), np.append(VX,np.ravel(VX1)), np.append(VY,np.ravel(VY1)), np.append(VZ,np.ravel(VZ1)), np.append(T,np.ravel(T1)), np.append(P,np.ravel(P1)) 
    
    XYZ = np.vstack([np.ravel(X),np.ravel(Y),np.ravel(Z)]).T
    
    # create Euler grid
    x , y , z = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.y0,params.y1,params.Ny),np.linspace(params.z0,params.z1,params.Nz),indexing='ij')
    x , y , z = np.ravel(x) , np.ravel(y) , np.ravel(z)
    xyz = np.vstack([x,y,z]).T
    
    # interpolate track points to Euler grid
    if params.interpolationMode == 'nd':
        vxInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), VX)
        vx = vxInterpolate(x,y,z)
        print('  vx - done')
        vyInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), VY)
        vy = vyInterpolate(x,y,z)
        print('  vy - done')
        vzInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), VZ)
        vz = vzInterpolate(x,y,z)
        print('  vz - done')
        TInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), T)
        Tint = TInterpolate(x,y,z)
        print('  T - done')
        PInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), P)
        p = PInterpolate(x,y,z)
        print('  p - done')
    elif params.interpolationMode == 'rbf':
        vx = RBFInterpolator(XYZ, VX, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  vx - done')
        vy = RBFInterpolator(XYZ, VY, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  vy - done')
        vz = RBFInterpolator(XYZ, VZ, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  vz - done')
        Tint = RBFInterpolator(XYZ, T, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  T - done')
        p = RBFInterpolator(XYZ, P, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  p - done')
    
    # correct Neumann boundaries
    p = p.reshape(params.Nx,params.Ny,params.Nz)
    p[:,:,0], p[:,:,-1] = p[:,:,1], p[:,:,-2]
    p[:,0,:], p[:,-1,:] = p[:,1,:], p[:,-2,:]
    p[0,:,:], p[-1,:,:] = p[1,:,:], p[-2,:,:]
    Tint = Tint.reshape(params.Nx,params.Ny,params.Nz)
    Tint[:,0,:], Tint[:,-1,:] = Tint[:,1,:], Tint[:,-2,:]
    Tint[0,:,:], Tint[-1,:,:] = Tint[1,:,:], Tint[-2,:,:]
    
    # save Euler grid    
    np.savetxt(params.field_path+'Euler/Euler_t_'+str(int(t))+'.txt', np.vstack([x,y,z,vx,vy,vz,Tint.ravel(),p.ravel()]).T,header='x, y, z, vx, vy, vz, T, p') 
    return x, y, z, vx, vy, vz, Tint, p 