'''

    This script contains interpolation functions.
    
'''


import numpy as np

from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


def InterpolateTracksToGrid(t,allTracks,params):
    # save Lagrange grid
    lagrangeGrid = np.asarray([track[np.argwhere(track[:,0]==t)[0][0]] for track in allTracks if np.argwhere(track[:,0]==t).size!=0]) 
    np.savetxt(params.field_path+'Lagrange/Lagrange_t_'+str(int(t))+'.txt',lagrangeGrid[:,1:10:],header='x, y, z, vx, vy, vz, ax, ay, az')
    
    # load track information 
    X , Y , Z = lagrangeGrid[::params.dN,1] , lagrangeGrid[::params.dN,2] , lagrangeGrid[::params.dN,3]
    VX , VY , VZ = lagrangeGrid[::params.dN,4] , lagrangeGrid[::params.dN,5] , lagrangeGrid[::params.dN,6]
    AX , AY , AZ = lagrangeGrid[::params.dN,7] , lagrangeGrid[::params.dN,8] , lagrangeGrid[::params.dN,9]
        
    # fill in zero boundary condition
        # y-z BC
    y_bc, z_bc = np.meshgrid(np.linspace(params.y0,params.y1,params.Ny),np.linspace(params.z0,params.z1,params.Nz))
    VX0 , VY0 , VZ0 = np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc) 
    VX1 , VY1 , VZ1 = np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc) 
    AX0 , AY0 , AZ0 = np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc)  
    AX1 , AY1 , AZ1 = np.zeros_like(y_bc) , np.zeros_like(y_bc) , np.zeros_like(y_bc)  
    X, Y, Z, VX, VY, VZ, AX, AY, AZ = np.append(X,np.ravel(params.x0*np.ones(params.Ny*params.Nz))), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX0)), np.append(VY,np.ravel(VY0)), np.append(VZ,np.ravel(VZ0)), np.append(AX,np.ravel(AX0)), np.append(AY,np.ravel(AY0)), np.append(AZ,np.ravel(AZ0))
    X, Y, Z, VX, VY, VZ, AX, AY, AZ = np.append(X,np.ravel(params.x1*np.ones(params.Ny*params.Nz))), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX1)), np.append(VY,np.ravel(VY1)), np.append(VZ,np.ravel(VZ1)), np.append(AX,np.ravel(AX1)), np.append(AY,np.ravel(AY1)), np.append(AZ,np.ravel(AZ1))     
        # x-z BC    
    x_bc, z_bc = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.z0,params.z1,params.Nz))
    VX0 , VY0 , VZ0 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc)  
    VX1 , VY1 , VZ1 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc)  
    AX0 , AY0 , AZ0 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc)  
    AX1 , AY1 , AZ1 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc)    
    X, Y, Z, VX, VY, VZ, AX, AY, AZ = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(params.y0*np.ones(params.Nx*params.Nz))), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX0)), np.append(VY,np.ravel(VY0)), np.append(VZ,np.ravel(VZ0)), np.append(AX,np.ravel(AX0)), np.append(AY,np.ravel(AY0)), np.append(AZ,np.ravel(AZ0))
    X, Y, Z, VX, VY, VZ, AX, AY, AZ = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(params.y1*np.ones(params.Nx*params.Nz))), np.append(Z,np.ravel(z_bc)), np.append(VX,np.ravel(VX1)), np.append(VY,np.ravel(VY1)), np.append(VZ,np.ravel(VZ1)), np.append(AX,np.ravel(AX1)), np.append(AY,np.ravel(AY1)), np.append(AZ,np.ravel(AZ1))   
        # x-y BC    
    x_bc, y_bc = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.y0,params.y1,params.Ny))
    VX0 , VY0 , VZ0 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) 
    VX1 , VY1 , VZ1 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc) 
    AX0 , AY0 , AZ0 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc)  
    AX1 , AY1 , AZ1 = np.zeros_like(x_bc) , np.zeros_like(x_bc) , np.zeros_like(x_bc)   
    X, Y, Z, VX, VY, VZ, AX, AY, AZ = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(params.z0*np.ones(params.Nx*params.Ny))), np.append(VX,np.ravel(VX0)), np.append(VY,np.ravel(VY0)), np.append(VZ,np.ravel(VZ0)), np.append(AX,np.ravel(AX0)), np.append(AY,np.ravel(AY0)), np.append(AZ,np.ravel(AZ0))
    X, Y, Z, VX, VY, VZ, AX, AY, AZ = np.append(X,np.ravel(x_bc)), np.append(Y,np.ravel(y_bc)), np.append(Z,np.ravel(params.z1*np.ones(params.Nx*params.Ny))), np.append(VX,np.ravel(VX1)), np.append(VY,np.ravel(VY1)), np.append(VZ,np.ravel(VZ1)), np.append(AX,np.ravel(AX1)), np.append(AY,np.ravel(AY1)), np.append(AZ,np.ravel(AZ1))   
    
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
        #axInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), AX)
        #ax = axInterpolate(x,y,z)
        #print('  ax - done')
        #ayInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), AY)
        #ay = ayInterpolate(x,y,z)
        #print('  ay - done')
        #azInterpolate = LinearNDInterpolator(list(zip(X, Y, Z)), AZ)
        #az = azInterpolate(x,y,z)
        #print('  az - done')
    elif params.interpolationMode == 'rbf':
        vx = RBFInterpolator(XYZ, VX, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  vx - done')
        vy = RBFInterpolator(XYZ, VY, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  vy - done')
        vz = RBFInterpolator(XYZ, VZ, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  vz - done')
        ax = RBFInterpolator(XYZ, AX, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  ax - done')
        ay = RBFInterpolator(XYZ, AY, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  ay - done')
        az = RBFInterpolator(XYZ, AZ, smoothing=params.smooth, kernel='cubic')(xyz)
        print('  az - done')
    
    # save Euler grid    
    #np.savetxt(params.field_path+'Euler/Euler_t_'+str(int(t))+'.txt', np.vstack([x,y,z,vx,vy,vz,ax,ay,az]).T,header='x, y, z, vx, vy, vz, ax, ay, az')   
    np.savetxt(params.field_path+'Euler/Euler_t_'+str(int(t))+'.txt', np.vstack([x,y,z,vx,vy,vz]).T,header='x, y, z, vx, vy, vz')   