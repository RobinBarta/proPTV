'''
    
    This script contains the probabilistic approximation method.

'''


import numpy as np 
import numpy.matlib as ml
from scipy.optimize import curve_fit


def GMM(t,X):
    N, ext = len(t), 3
    # set center of Gaussians and kernel sizes
    centers = np.array([np.linspace(0-ext,1+ext,N)])
    kernel_size = ((centers[0,1]-centers[0,0]))**2
    # estimate Gaussians variables  and derivatives based on dimensionless time z
    z = np.linspace(0, 1, len(t))
    x = (ml.repmat(z, N, 1) - ml.repmat(centers.T, 1, len(t)))
    b = np.exp(- x**2 / (2 * kernel_size))
    b_dt = b * (- x / kernel_size)
    b_dt_dt = (b * (- x / kernel_size)**2) - (b / kernel_size)  
    sum_b = ml.repmat((np.sum(b, axis=0)), N, 1)
    sum_b_dt = ml.repmat((np.sum(b_dt, axis=0)), N, 1)
    sum_b_dt_dt = ml.repmat((np.sum(b_dt_dt, axis=0)), N, 1)
    # get basis functions
    psi_X = (b / sum_b).T
    psi_V = ((b_dt * sum_b - b * sum_b_dt) / (sum_b**2) * (z[1] - z[0])).T
    psi_A = ((((b_dt_dt * sum_b - b * sum_b_dt_dt)*sum_b**2) - ((b_dt*sum_b-b*sum_b_dt)*(2*sum_b*sum_b_dt))) / (sum_b**4) * (z[1] - z[0])**2).T
    # get weights of basis functions
    w = np.linalg.solve(np.matmul(psi_X.T,psi_X) + np.eye(psi_X.shape[1])*1e-10, np.matmul(psi_X.T,X))
    return w, psi_X, psi_V, psi_A

def Approximate(t, w, psi_X, psi_V, psi_A):
    X = np.matmul(psi_X,w)
    V = np.matmul(psi_V,w)
    A = np.matmul(psi_A,w)
    return X, V, A

def Uncertanty(t, tracks_X, X, V, A, w, psi_X, psi_V, psi_A, dim):
    w_mean = np.zeros([len(t),dim])
    std_X, std_V, std_A = np.zeros([len(t),dim]), np.zeros([len(t),dim]), np.zeros([len(t),dim])
    for d in range(dim):
        Omega_d = np.asarray([w[:,d]]+[GMM(t,pos[:,d])[0] for pos in tracks_X]).T
        w_mean[:,d] = np.mean(Omega_d,axis=1)
        Cov = np.cov(Omega_d)
        std_X[:,d] = np.asarray([np.sqrt(np.matmul(psi_X[i,:], np.matmul(Cov,psi_X[i,:].T))) for i in range(len(psi_X))])
        std_X[:,d] =  np.max(np.abs(np.diff(X[:,d]))) * (std_X[:,d]/np.max(std_X[:,d]))
        std_X[0,d], std_X[-1,d] = np.max(std_X[:,d]), np.max(std_X[:,d])
        std_V[:,d] = np.asarray([np.sqrt(np.matmul(psi_V[i,:], np.matmul(Cov,psi_V[i,:].T))) for i in range(len(psi_V))])
        std_V[:,d] =  np.max(np.abs(np.diff(V[:,d]))) * (std_V[:,d]/np.max(std_V[:,d]))
        std_V[0,d], std_V[-1,d] = np.max(std_V[:,d]), np.max(std_V[:,d])
        std_A[:,d] = np.asarray([np.sqrt(np.matmul(psi_A[i,:], np.matmul(Cov,psi_A[i,:].T))) for i in range(len(psi_A))])
        std_A[:,d] =  np.max(np.abs(np.diff(A[:,d]))) * (std_A[:,d]/np.max(std_A[:,d]))
        std_A[0,d], std_A[-1,d] = np.max(std_A[:,d]), np.max(std_A[:,d])
    return np.squeeze(w_mean), np.squeeze(std_X), np.squeeze(std_V), np.squeeze(std_A)

def Predict(t, X, V, A):
    return X[-1]+V[-1]*np.sign(np.diff(t)[0]), V[-1]+A[-1]*np.sign(np.diff(t)[0]), A[-1]+(A[-1]-A[-2])
    
def TrackingProbability(x,X,std_X,dim):
    return np.exp(-(np.linalg.norm(x-X)**2/(2*np.linalg.norm(std_X)**2)))

def Gauss(x,A,B,C):
    return A*(np.exp(-(x-B)**2/(2*C**2)))
def dGauss(x,A,B,C):
    return A*(np.exp(-(x-B)**2/(2*C**2))) * (-(x-B)/(C**2))
def ddGauss(x,A,B,C):
    return (A*(np.exp(-(x-B)**2/(2*C**2))) * (-(x-B)/(C**2))**2) + (A*(np.exp(-(x-B)**2/(2*C**2))) * (-1/(C**2)))
def Predict_HD(t, t_step, N_t, w, psi_X):
    w = w.reshape(len(w), -1)
    # estimate high definition time step
    delta_t = np.diff(t)[0]
    t_HD = np.linspace( t[0]-(t_step*delta_t), t[-1]+(t_step*delta_t), (len(t)+(2*t_step))*N_t )
    X_HD, V_HD, A_HD = np.zeros([len(t_HD),w.shape[-1]]), np.zeros([len(t_HD),w.shape[-1]]), np.zeros([len(t_HD),w.shape[-1]])
    # fit basis functions
    for d in range(len(w[0,:])):
        BasisF = []
        for i in range(len(w)):
            G = psi_X[:,i]*w[i,d]
            popt,pcov = curve_fit(Gauss,t,G,p0=[np.sign(G[np.argmax(np.abs(G))])*np.max(np.abs(G)),t[np.argmax(np.abs(G))],1.0],bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,50]))
            BasisF.append(popt)
        # predict pos, vel , acc
        X_HD[:,d] = np.sum([Gauss(t_HD,*p) for p in BasisF],axis=0)
        V_HD[:,d] = np.sum([dGauss(t_HD,*p) for p in BasisF],axis=0)
        A_HD[:,d] = np.sum([ddGauss(t_HD,*p) for p in BasisF],axis=0)
    return t_HD, np.squeeze(X_HD), np.squeeze(V_HD), np.squeeze(A_HD)