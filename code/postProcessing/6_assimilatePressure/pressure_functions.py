'''

    This script contains functions for the pressure estimation

'''


import numpy as np


def EstimateTerms(u,v,w,params):
    convection_u, convection_v, convection_w = np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)
    diffusion_u, diffusion_v, diffusion_w = np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)
    '''
    convection_u[1:-1,1:-1,1:-1] = (u[1:-1,1:-1,1:-1]*(u[2:,1:-1,1:-1] - u[1:-1,1:-1,1:-1])/params.dx) + (v[1:-1,1:-1,1:-1]*(u[1:-1,2:,1:-1] - u[1:-1,1:-1,1:-1])/params.dy) + (w[1:-1,1:-1,1:-1]*(u[1:-1,1:-1,2:] - u[1:-1,1:-1,1:-1])/params.dz)
    convection_v[1:-1,1:-1,1:-1] = (u[1:-1,1:-1,1:-1]*(v[2:,1:-1,1:-1] - v[1:-1,1:-1,1:-1])/params.dx) + (v[1:-1,1:-1,1:-1]*(v[1:-1,2:,1:-1] - v[1:-1,1:-1,1:-1])/params.dy) + (w[1:-1,1:-1,1:-1]*(v[1:-1,1:-1,2:] - v[1:-1,1:-1,1:-1])/params.dz)
    convection_w[1:-1,1:-1,1:-1] = (u[1:-1,1:-1,1:-1]*(w[2:,1:-1,1:-1] - w[1:-1,1:-1,1:-1])/params.dx) + (v[1:-1,1:-1,1:-1]*(w[1:-1,2:,1:-1] - w[1:-1,1:-1,1:-1])/params.dy) + (w[1:-1,1:-1,1:-1]*(w[1:-1,1:-1,2:] - w[1:-1,1:-1,1:-1])/params.dz)
    diffusion_u[2:-2,2:-2,2:-2] = np.sqrt(6.9/1E10) * ( (((u[2:,1:-1,1:-1] - u[1:-1,1:-1,1:-1])/params.dx)[2:,1:-1,1:-1]-((u[2:,1:-1,1:-1] - u[1:-1,1:-1,1:-1])/params.dx)[1:-1,1:-1,1:-1])/params.dx + 
                                                        (((u[1:-1,2:,1:-1] - u[1:-1,1:-1,1:-1])/params.dy)[1:-1,2:,1:-1]-((u[1:-1,2:,1:-1] - u[1:-1,1:-1,1:-1])/params.dy)[1:-1,1:-1,1:-1])/params.dy +
                                                        (((u[1:-1,1:-1,2:] - u[1:-1,1:-1,1:-1])/params.dz)[1:-1,1:-1,2:]-((u[1:-1,1:-1,2:] - u[1:-1,1:-1,1:-1])/params.dz)[1:-1,1:-1,1:-1])/params.dz )
    diffusion_v[2:-2,2:-2,2:-2] = np.sqrt(6.9/1E10) * ( (((v[2:,1:-1,1:-1] - v[1:-1,1:-1,1:-1])/params.dx)[2:,1:-1,1:-1]-((v[2:,1:-1,1:-1] - v[1:-1,1:-1,1:-1])/params.dx)[1:-1,1:-1,1:-1])/params.dx + 
                                                        (((v[1:-1,2:,1:-1] - v[1:-1,1:-1,1:-1])/params.dy)[1:-1,2:,1:-1]-((v[1:-1,2:,1:-1] - v[1:-1,1:-1,1:-1])/params.dy)[1:-1,1:-1,1:-1])/params.dy +
                                                        (((v[1:-1,1:-1,2:] - v[1:-1,1:-1,1:-1])/params.dz)[1:-1,1:-1,2:]-((v[1:-1,1:-1,2:] - v[1:-1,1:-1,1:-1])/params.dz)[1:-1,1:-1,1:-1])/params.dz )
    diffusion_w[2:-2,2:-2,2:-2] = np.sqrt(6.9/1E10) * ( (((w[2:,1:-1,1:-1] - w[1:-1,1:-1,1:-1])/params.dx)[2:,1:-1,1:-1]-((w[2:,1:-1,1:-1] - w[1:-1,1:-1,1:-1])/params.dx)[1:-1,1:-1,1:-1])/params.dx + 
                                                        (((w[1:-1,2:,1:-1] - w[1:-1,1:-1,1:-1])/params.dy)[1:-1,2:,1:-1]-((w[1:-1,2:,1:-1] - w[1:-1,1:-1,1:-1])/params.dy)[1:-1,1:-1,1:-1])/params.dy +
                                                        (((w[1:-1,1:-1,2:] - w[1:-1,1:-1,1:-1])/params.dz)[1:-1,1:-1,2:]-((w[1:-1,1:-1,2:] - w[1:-1,1:-1,1:-1])/params.dz)[1:-1,1:-1,1:-1])/params.dz )
    '''
    for i in range(2,params.Nx+1):
        for j in range(1,params.Ny+1):
            for k in range(1,params.Nz+1):
                ue = 0.5*(u[i+1,j,k] + u[i,j,k])
                uw = 0.5*(u[i,j,k]   + u[i-1,j,k])            
                un = 0.5*(u[i,j+1,k] + u[i,j,k])
                us = 0.5*(u[i,j,k] + u[i,j-1,k])             
                uf = 0.5*(u[i,j,k+1] + u[i,j,k])
                ub = 0.5*(u[i,j,k] + u[i,j,k-1])           
                vn = 0.5*(v[i,j+1,k] + v[i,j,k])
                vs = 0.5*(v[i,j,k] + v[i,j-1,k])      
                wf = 0.5*(w[i,j,k+1] + w[i,j,k])
                wb = 0.5*(w[i,j,k] + w[i,j,k-1])
                # convection = d(uu)/dx + d(vu)/dy + d(wu)/dz
                convection_u[i,j,k] = (ue*ue - uw*uw)/params.dx + (vn*un - vs*us)/params.dy + (wf*uf - wb*ub)/params.dz
                # diffusion = d2u/dx2 + d2u/dy2 + d2u/dz2
                diffusion_u[i,j,k] = np.sqrt(params.Pr/params.Ra)*( (u[i+1,j,k] - 2.0*u[i,j,k] + u[i-1,j,k])/params.dx/params.dx + 
                                                                    (u[i,j+1,k] - 2.0*u[i,j,k] + u[i,j-1,k])/params.dy/params.dy +
                                                                    (u[i,j,k+1] - 2.0*u[i,j,k] + u[i,j,k-1])/params.dz/params.dz )
    for i in range(1,params.Nx+1):
        for j in range(2,params.Ny+1):
            for k in range(1,params.Nz+1):
                ve = 0.5*(v[i+1,j,k] + v[i,j,k])
                vw = 0.5*(v[i,j,k]   + v[i-1,j,k])            
                vn = 0.5*(v[i,j+1,k] + v[i,j,k])
                vs = 0.5*(v[i,j,k] + v[i,j-1,k])             
                vf = 0.5*(v[i,j,k+1] + v[i,j,k])
                vb = 0.5*(v[i,j,k] + v[i,j,k-1])         
                ue = 0.5*(u[i+1,j,k] + u[i,j,k])
                uw = 0.5*(u[i,j,k]   + u[i-1,j,k])   
                wf = 0.5*(w[i,j,k+1] + w[i,j,k])
                wb = 0.5*(w[i,j,k] + w[i,j,k-1])
                # convection = d(uv)/dx + d(vv)/dy + d(wv)/dz
                convection_v[i,j,k] = (ue*ve - uw*vw)/params.dx + (vn*vn - vs*vs)/params.dy + (wf*vf - wb*vb)/params.dz
                # diffusion = d2v/dx2 + d2v/dy2 + d2v/dz2
                diffusion_v[i,j,k] = np.sqrt(params.Pr/params.Ra)*( (v[i+1,j,k] - 2.0*v[i,j,k] + v[i-1,j,k])/params.dx/params.dx + 
                                                                    (v[i,j+1,k] - 2.0*v[i,j,k] + v[i,j-1,k])/params.dy/params.dy +
                                                                    (v[i,j,k+1] - 2.0*v[i,j,k] + v[i,j,k-1])/params.dz/params.dz )
    for i in range(1,params.Nx+1):
        for j in range(1,params.Ny+1):
            for k in range(2,params.Nz+1):
                we = 0.5*(w[i+1,j,k] + w[i,j,k])
                ww = 0.5*(w[i,j,k]   + w[i-1,j,k])            
                wn = 0.5*(w[i,j+1,k] + w[i,j,k])
                ws = 0.5*(w[i,j,k] + w[i,j-1,k])             
                wf = 0.5*(w[i,j,k+1] + w[i,j,k])
                wb = 0.5*(w[i,j,k] + w[i,j,k-1])         
                ue = 0.5*(u[i+1,j,k] + u[i,j,k])
                uw = 0.5*(u[i,j,k]   + u[i-1,j,k])          
                vn = 0.5*(v[i,j+1,k] + v[i,j,k])
                vs = 0.5*(v[i,j,k] + v[i,j-1,k]) 
                # convection = d(uw)/dx + d(vw)/dy + d(ww)/dz
                convection_w[i,j,k] = (ue*we - uw*ww)/params.dx + (vn*wn - vs*ws)/params.dy + (wf*wf - wb*wb)/params.dz
                # diffusion = d2v/dx2 + d2v/dy2 + d2v/dz2
                diffusion_w[i,j,k] = np.sqrt(params.Pr/params.Ra)*( (w[i+1,j,k] - 2.0*w[i,j,k] + w[i-1,j,k])/params.dx/params.dx + 
                                                                    (w[i,j+1,k] - 2.0*w[i,j,k] + w[i,j-1,k])/params.dy/params.dy +
                                                                    (w[i,j,k+1] - 2.0*w[i,j,k] + w[i,j,k-1])/params.dz/params.dz )
    #'''
    return convection_u, convection_v, convection_w, diffusion_u, diffusion_v, diffusion_w
