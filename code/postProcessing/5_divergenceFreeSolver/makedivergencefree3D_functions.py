'''

    This script contains functions to solves the pressure poisson equation on an euler grid to make field divergence free.
    
'''


import numpy as np
import scipy.sparse.linalg


def LoadEuler3D(t,params):
    data = np.loadtxt(params.case_name+"/output/"+params.runname+"/fields/Euler/Euler_t_{time}.txt".format(time=int(t)))
    x, y, z, vx, vy, vz = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
    xyz, uvw = np.vstack([x,y,z]).T, np.vstack([vx,vy,vz]).T
    xc, yc, zc = xyz[:,0].reshape(params.Nz,params.Ny,params.Nx) , xyz[:,1].reshape(params.Nz,params.Ny,params.Nx) , xyz[:,2].reshape(params.Nz,params.Ny,params.Nx) 
    vx_new , vy_new , vz_new = uvw[:,0].reshape(params.Nz,params.Ny,params.Nx) , uvw[:,1].reshape(params.Nz,params.Ny,params.Nx) , uvw[:,2].reshape(params.Nz,params.Ny,params.Nx) 
    return xc , yc , zc , vx_new , vy_new, vz_new

def Initialise3DVariables(vx,vy,vz,params):
    u , v , w , p = np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2])
    u[1:-1,1:-1,1:-1] , v[1:-1,1:-1,1:-1] , w[1:-1,1:-1,1:-1] = vx, vy, vz
    # define correction field
    u_sol , v_sol , w_sol = np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2]) , np.zeros([params.Nz+2,params.Ny+2,params.Nx+2])
    return u, v, w, p, u_sol, v_sol , w_sol

def Divergence3D(u,v,w,params):
    div = (u[2:,1:-1,1:-1]-u[1:-1,1:-1,1:-1])/params.dx  + (v[1:-1,2:,1:-1]-v[1:-1,1:-1,1:-1])/params.dy  + (w[1:-1,1:-1,2:]-w[1:-1,1:-1,1:-1])/params.dz
    return div

def Get3DSparseMatrix(params):
    # set left right top bot front back coefs
    Aw , Ae , An , As , Af, Ab = 1.0/params.dx/params.dx*np.ones([params.Nz,params.Ny,params.Nx]) , 1.0/params.dx/params.dx*np.ones([params.Nz,params.Ny,params.Nx]) , 1.0/params.dy/params.dy*np.ones([params.Nz,params.Ny,params.Nx]) , 1.0/params.dy/params.dy*np.ones([params.Nz,params.Ny,params.Nx]) , 1.0/params.dz/params.dz*np.ones([params.Nz,params.Ny,params.Nx]) , 1.0/params.dz/params.dz*np.ones([params.Nz,params.Ny,params.Nx])
    Ae[:,:,-1] , Aw[:,:,0] , An[:,-1,:] , As[:,0,:] , Af[-1,:,:] , Ab[0,:,:] = 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 
    A0 = -(Aw + Ae + An + As + Af + Ab)
    d0 = A0.reshape(params.Nx*params.Ny*params.Nz)
    de = Ae.reshape(params.Nx*params.Ny*params.Nz)[:-1]
    dw = Aw.reshape(params.Nx*params.Ny*params.Nz)[1:]
    dn = An.reshape(params.Nx*params.Ny*params.Nz)[:-params.Nx]
    ds = As.reshape(params.Nx*params.Ny*params.Nz)[params.Nx:]
    df = Af.reshape(params.Nx*params.Ny*params.Nz)[:-params.Nx*params.Ny]
    db = Ab.reshape(params.Nx*params.Ny*params.Nz)[params.Nx*params.Ny:]
    A = scipy.sparse.diags([d0, de, dw, dn, ds, df, db], [0, 1, -1, params.Nx, -params.Nx, params.Nx*params.Ny, -params.Nx*params.Ny], format='csr')
    #plt.spy(A)
    #print(A.toarray())
    return A

def HelmholtzHodgeCorrection(u_sol,v_sol,w_sol,u,v,w,p,params):
    u_sol[2:-1,1:-1,1:-1] = u[2:-1,1:-1,1:-1] - (p[2:-1,1:-1,1:-1] - p[1:-2,1:-1,1:-1])/params.dx 
    v_sol[1:-1,2:-1,1:-1] = v[1:-1,2:-1,1:-1] - (p[1:-1,2:-1,1:-1] - p[1:-1,1:-2,1:-1])/params.dy
    w_sol[1:-1,1:-1,2:-1] = w[1:-1,1:-1,2:-1] - (p[1:-1,1:-1,2:-1] - p[1:-1,1:-1,1:-2])/params.dz
    return u_sol, v_sol, w_sol


#%%     unused


def StaggeredGrid3D(params):
    # cell centered coordinates
    x = np.linspace(params.x0+params.dx/2.0, params.x1-params.dx/2.0, params.Nx, endpoint=True)
    y = np.linspace(params.y0+params.dy/2.0, params.y1-params.dy/2.0, params.Ny, endpoint=True)
    z = np.linspace(params.z0+params.dz/2.0, params.z1-params.dz/2.0, params.Nz, endpoint=True)
    yc, zc, xc = np.meshgrid(x,y,z)
    # x-staggered coordinates
    xs = np.linspace(params.x0, params.x1, params.Nx+1, endpoint=True)
    yu, zu, xu = np.meshgrid(xs, y, z)
    # y-staggered coordinates
    ys = np.linspace(params.y0, params.y1, params.Ny+1, endpoint=True)
    yv, zv, xv = np.meshgrid(x, ys, z)
    # z-staggered coordinates
    zs = np.linspace(params.z0, params.z1, params.Nz+1, endpoint=True)
    yw, zw, xw = np.meshgrid(x, y, zs)
    return xc , yc , zc , xu , yu , zu , xv , yv , zv , xw , yw , zw