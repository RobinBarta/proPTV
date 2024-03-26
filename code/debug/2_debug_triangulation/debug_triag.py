'''

    Debug triangulation.
    
'''


import os, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

shutil.copy('../../main/functions/triangulation.py', 'functions/')
shutil.copy('../../main/functions/soloff.py', 'functions/')
from functions.triangulation import *

os.chdir('../../../data')


# %%

class TriagParameter():
    case_name, runname, Zeros = "45000", "run1", 5
    cams, t = [0,1,2,3], 1
    depthsaxis = [0,0,0,0]
    
    d, mode = 0.0075, 'load'
    n = 0
    
    Vmin , Vmax = [0,0,0], [1,1,1]   
    startCamsUsedForPermutation, numberOfPermutations = [0], 1  
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling = 1, 4, 3, 3, 1.0, 0.0     
    
# %%


def main(): 
    # load parameter
    params = TriagParameter()
    params.case_path = params.case_name+"/" 
    params.calibration_path = params.case_path + "input/calibration/c{cam}/soloff_c{cam}{xy}.txt" 
    params.imgPoints_path = params.case_path + "input/particle_lists/c{cam}/c{cam}_{timeString}.txt"  
    params.triangulation_path = "../code/debugging/2_debug_triangulation/output"  
    
    # load calibration
    ax = [np.loadtxt(params.case_name+'/input/calibration/c{cam}/soloff_c{cam}{xy}.txt'.format(cam=cam,xy="x"),delimiter=',') for cam in params.cams]
    ay = [np.loadtxt(params.case_name+'/input/calibration/c{cam}/soloff_c{cam}{xy}.txt'.format(cam=cam,xy="y"),delimiter=',') for cam in params.cams]
    # load img points
    imgPs = np.asarray([np.loadtxt(params.case_name+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'.format(cam=c,time=str(params.t).zfill(params.Zeros)),skiprows=1) for c in params.cams],dtype=object)
    
    # do triangulation
    print('Triangulation:')
    if params.mode == 'load':
        Triag = np.loadtxt(params.case_name+"/output/"+params.runname+"/triangulation/Points_{time}.txt".format(time=str(params.t).zfill(params.Zeros)),skiprows=1)
    else:
        Triag , imgPs_new = Triangulate3DPoints(imgPs,params.t,ax,ay,params)
    print(' triangulated ' + str(len(Triag)) + ' points from ' + str([len(ImgP) for ImgP in imgPs]) + ' image points\n' )
    
    # debug triangulation with ground truth
    if os.path.isfile(params.case_name+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros))):
        Ps = np.loadtxt(params.case_name+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros)),skiprows=1)[:,1::]
        counts, dels, dels_Triag, wrongID = 0, [], [] ,[]
        for i , p in enumerate(tqdm(Triag,desc='Debug Triangulation: ',position=0,leave=True,delay=0.5)):
            positions, imgpoints = p[:3:], p[4::]
            dP = np.linalg.norm(positions-Ps[:,:3:],axis=1)
            ID = np.argmin(dP)
            if dP[ID]<params.d:
                if not ID in dels:
                    counts += 1
                dels.append(ID)
                dels_Triag.append(i)
            else:
                wrongID.append(dP[ID])
        print(len(dels))
        print(len(np.unique(dels)))
        print(len(dels_Triag))
        print(len(np.unique(dels_Triag)))
        Ps_del = np.delete(Ps,dels,axis=0)
        Triag_del = np.delete(Triag,dels_Triag,axis=0)
        T1 = Triag[np.asarray(dels_Triag)]
        T2 = Ps[np.asarray(dels)]
        print('\n found ' + str(counts) + ' / ' + str(len(Ps)) + ' ( '+ str(round(counts/len(Ps)*100,2)) +' % ) true points')
        print(' found ' + str(len(Triag)-counts) + ' / ' + str(len(Triag)) + ' ( '+ str(round((len(Triag)-counts)/len(Triag)*100,2)) +' % ) wrong points')
        
        plt.figure()
        plt.hist(wrongID,bins=50)
        plt.show()
        
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.scatter(T1[:,0],T1[:,1],T1[:,2],color='red',s=0.8)
        axis.scatter(T2[:,0],T2[:,1],T2[:,2],color='green',s=0.8)
        plt.show()
        
        # plot points
        fig, axis = plt.subplots(1,2,figsize=(16,8),subplot_kw=dict(projection='3d'))
        axis[0].scatter(Triag[:,0],Triag[:,1],Triag[:,2],color='red',s=0.8)
        axis[0].scatter(Ps[:,0],Ps[:,1],Ps[:,2],color='green',s=0.8)
        axis[1].scatter(Triag_del[:,0],Triag_del[:,1],Triag_del[:,2],color='red',s=0.8)
        axis[1].scatter(Ps_del[:,0],Ps_del[:,1],Ps_del[:,2],color='green',s=0.8)
        plt.show()
    else:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.scatter(Triag[:,0],Triag[:,1],Triag[:,2],color='red',s=0.1)
        plt.show()
        
    # plot projection onto images
    fig, axis = plt.subplots(1,4,figsize=(16,4),sharex=True,sharey=True)
    for i,c in enumerate(params.cams):
        x, y = Triag[params.n,4+int(2*i)], Triag[params.n,4+int(2*i)+1]
        img = cv2.imread(params.case_name+'/input/raw_images/c{cam}/c{cam}_{time}.tif'.format(cam=c,time=str(params.t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) 
        axis[i].imshow(img,cmap='gray',vmax=2000)
        axis[i].plot(x,y,'o',c='red')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()