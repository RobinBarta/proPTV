'''

    This script contains the soloff polynomial and the optimization function during extending and triangulation since it is based on the soloff polynom.

'''


import numpy as np


def F(XYZ, a):
    '''
        soloff polynom
    '''
    if len(XYZ.shape) == 1:
        X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    else:
        X , Y , Z = XYZ[:,0] , XYZ[:,1] , XYZ[:,2]
    return ( a[0] 
                + X * ( X*(a[9]*X+a[11]*Y+a[14]*Z+a[4]) + a[13]*Y*Z + a[6]*Y + a[7]*Z + a[1] ) 
                + Y * ( Y*(a[12]*X+a[10]*Y+a[15]*Z+a[5]) + a[8]*Z + a[2] ) 
                + Z * ( Z*(a[17]*X+a[18]*Y+a[16]) + a[3] ) ) 

def dFdx(XYZ, a):
    '''
        derivative of soloff polynom by x
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return (3 * a[9] * pow(X, 2) + 2 * a[11] * X * Y + 2 * a[14] * X * Z + 2 * a[4] * X 
            + a[12] * pow(Y, 2) + a[13] * Y * Z + a[6] * Y
            + a[17] * pow(Z, 2) + a[7] * Z + a[1])
            
def dFdy(XYZ, a):
    '''
        derivative of soloff polynom by y
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return (a[11] * pow(X, 2) + 2 * a[12] * X * Y + a[13] * X * Z + a[6] * X 
            + 3 * a[10] * pow(Y, 2) + 2 * a[15] * Y * Z + 2 * a[5] * Y 
            + a[18] * pow(Z, 2) + a[8] * Z + a[2])
            
def dFdz(XYZ, a):
    '''
        derivative of soloff polynom by z
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return  (2 * Z * (a[17] * X + a[18] * Y + a[16]) 
            + X * (a[14] * X + a[13] * Y + a[7]) 
            + Y * (a[15] * Y + a[8]) + a[3])

def Cost_Function(setP,P,ax,ay):
    '''
        calculates the cost function per active camera
        (F(P) - camP) for each active cam
        difference between particle camera position and reprojected camera position
    '''
    cost = np.ravel( [[F(P,ax[i])-setP[i][0] , F(P,ay[i])-setP[i][1]] for i in np.arange(len(ax))] )
    return cost

def Jacobian_Soloff(P,ax,ay):
    '''
        calculates the Jacobian matrix of the soloff polynom for gradient descent algorithm
    '''
    jac = [ [[dFdx(P,ax[i]) , dFdy(P,ax[i]) , dFdz(P,ax[i])] , [dFdx(P,ay[i]) , dFdy(P,ay[i]) , dFdz(P,ay[i])]] for i in np.arange(len(ax)) ]
    J = np.asarray(sum(jac,[])) # [j for i in jac for j in i]
    return J

def NewtonSoloff_Triangulation(setP, ax, ay, params):
    '''
        Newton Soloff Algorithm to triangulate particle positions
    '''
    setP = np.asarray(setP)
    foundSetPoints = np.argwhere(np.isnan(setP[:,0])==False)
    setP , aX , aY = setP[foundSetPoints[:,0]] , np.asarray(ax)[foundSetPoints[:,0]] , np.asarray(ay)[foundSetPoints[:,0]]
    P = np.array([ (params.Vmax[0]+params.Vmin[0])/2 , (params.Vmax[1]+params.Vmin[1])/2 , (params.Vmax[2]+params.Vmin[2])/2 ])
    for i in range(3):
        P += np.linalg.lstsq(Jacobian_Soloff(P, aX, aY),-Cost_Function(setP, P, aX, aY),rcond=None)[0]
    costsP = np.linalg.norm(Cost_Function(setP, P, aX, aY).reshape(len(aX),2),axis=1) # cost per cam
    return P, costsP

def NewtonSoloff_Extend(setP, P_predict, aX, aY):
    '''
        Newton Soloff Algorithm to correct 3D particle position during extending
    '''
    P = np.zeros(4)
    P[:3:] = P_predict
    # Optimize Particle Position
    for i in range(3):
        P[:3:] += np.linalg.lstsq( Jacobian_Soloff(P[:3:], aX, aY) , -Cost_Function(setP, P[:3:], aX, aY) ,rcond=None)[0]
    P[-1] = np.mean(np.linalg.norm(Cost_Function(setP,P[:3:],aX,aY).reshape(2*len(aX),1)))
    return P