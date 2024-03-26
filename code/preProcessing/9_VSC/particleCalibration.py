'''

    Perfrom calibration based on particle images.
    
        ToDo: correct NaN values when activeMatches_triag<len(cams)
    
'''


import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

shutil.copy('../../main/functions/triangulation.py', 'functions/')
shutil.copy('../../main/functions/soloff.py', 'functions/')
shutil.copy('../../preProcessing/7_calibration/calibration_functions.py', 'functions/')
from functions.triangulation import *
from functions.calibration_functions import *

os.chdir('../../../data/')


# %%

class VSCParameter():    
    case_name, Zeros = 'rbc_300mm', 7
    cams = [0,1,2,3]
    t0, t1, dt = 2, 2, 1    
    
    Vmin , Vmax = [0,0,0] , [300,300,300]
    startCamsUsedForPermutation, numberOfPermutations = [0], 1 
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling = 1, 4, 3, 3, 1.0, 0.0
    
    runs = 1
    subVs_x = [5,7,11]
    subVs_y = [4,4,5]
    subVs_z = [5,7,11]
    
# %%


def main(): 
    # load parameter
    params = VSCParameter() 
    params.particle_path = params.case_name+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'
    params.marker_path = params.case_name+"/input/calibration_images/markers_c{cam}.txt"
    params.triangulation_path = "../code/preProcessing/9_VSC/output"  
    params.calibration_output =  "../code/preProcessing/9_VSC/parameter/c{cam}/soloff_c{cam}{xy}" 
    # load marker points
    markerPoints = [np.loadtxt(params.marker_path.format(cam=str(c)),skiprows=1) for c in params.cams]  

    # initial calibration by least square calibration method
    ax , ay = [] , []
    for i,c in enumerate(tqdm(params.cams,desc='Inital Calibration: ',leave=True,position=0)): 
        sx , sy = Calibration(markerPoints[i], [np.zeros(19),np.zeros(19)])
        ax.append(sx), ay.append(sy)
        # save initial calibration
        np.savetxt(params.calibration_output.format(cam=c,xy='x')+'_run0.txt',sx)
        np.savetxt(params.calibration_output.format(cam=c,xy='y')+'_run0.txt',sy)
    params.calibration_path = "../code/preProcessing/9_VSC/parameter/c{cam}/soloff_c{cam}{xy}_run0.txt"
    
    # estimate disparitiy
    d = [ np.vstack([ markerPoints[i][:,0]-Soloff(markerPoints[i][:,2::],ax[i]) , markerPoints[i][:,1]-Soloff(markerPoints[i][:,2::],ay[i]) ]).T for i,c in enumerate((range(len(params.cams))))]
    [print(' c{cam} mean calib error: '.format(cam=c),np.mean(np.linalg.norm(d[i],axis=1)),' +- ',np.std(np.linalg.norm(d[i],axis=1))) for i,c in enumerate(range(len(params.cams)))]
    
    # runs
    times = np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int)[::params.dt]
    print('\nStart Soloff particle calibration:')
    for run in range(1,1+params.runs):
        print(' run ' + str(run) + ':')
        for v in range(len(params.subVs_x)):
            print('  subvolumes: ' + str([params.subVs_x[v],params.subVs_y[v],params.subVs_z[v]]))
            # define arrays that split the measurement volume in the subvolumes
            V_x = np.linspace(params.Vmin[0],params.Vmax[0],params.subVs_x[v]+1)
            V_y = np.linspace(params.Vmin[1],params.Vmax[1],params.subVs_y[v]+1)
            V_z = np.linspace(params.Vmin[2],params.Vmax[2],params.subVs_z[v]+1)
            # triangulate particles
            P_clouds = np.empty([0,12])
            for t in times:
                print('   t = ' + str(t) + ':')
                ImgPoints = [np.loadtxt(params.particle_path.format(cam=cam,time=str(t).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
                print('   found particles per cam: ' , str([len(imgP) for imgP in ImgPoints]))
                TriagPoints , ImgPoints = Triangulate3DPoints(ImgPoints,t,ax,ay,params)
                #TriagPoints = np.loadtxt(params.triangulation_path+'/Points_00000.txt')
                P_clouds = np.append(P_clouds,TriagPoints,axis=0)
                print('   triangulated ' + str(len(TriagPoints)) + ' particles')
            print('   finally triangulated ' + str(len(P_clouds)) + ' particles')
            # for each camera - calcualte disparity maps and correct marker lists
            for i,c in enumerate(range(len(params.cams))):
                # get the marker points of the current camera
                x_corr , y_corr , XYZ_corr = np.empty([0]) , np.empty([0]) , np.empty([0,3])
                x, y, XYZ = markerPoints[i][:,0] , markerPoints[i][:,1] , markerPoints[i][:,2::]
                # go in every sub volume
                for xi in np.arange(params.subVs_x[v]):
                    for yj in np.arange(params.subVs_y[v]):
                        for zk in np.arange(params.subVs_z[v]):
                            # collect each 3D point inside the subvolume
                            IDs_P = np.argwhere( (V_x[0+xi] <= P_clouds[:,0]) & (P_clouds[:,0] <= V_x[1+xi]) & 
                                                 (V_y[0+yj] <= P_clouds[:,1]) & (P_clouds[:,1] <= V_y[1+yj]) & 
                                                 (V_z[0+zk] <= P_clouds[:,2]) & (P_clouds[:,2] <= V_z[1+zk]) )[:,0]
                            # collect each marker point inside the subvolume
                            IDs_marker = np.argwhere( (V_x[0+xi] <= XYZ[:,0]) & (XYZ[:,0] <= V_x[1+xi]) & 
                                                      (V_y[0+yj] <= XYZ[:,1]) & (XYZ[:,1] <= V_y[1+yj]) & 
                                                      (V_z[0+zk] <= XYZ[:,2]) & (XYZ[:,2] <= V_z[1+zk]) )[:,0]
                            
                            #axis = plt.figure().add_subplot(projection='3d')
                            #axis.scatter(P_clouds[IDs_P,0],P_clouds[IDs_P,1],P_clouds[IDs_P,2],c='black')
                            #axis.scatter(XYZ[IDs_marker,0],XYZ[IDs_marker,1],XYZ[IDs_marker,2],c='red')
                            #axis.set_xlim(0,300), axis.set_ylim(0,300), axis.set_zlim(0,300)
                            #plt.show()
                            #d = np.vstack([ P_clouds[IDs_P,4+int(2*i)]-Soloff(P_clouds[IDs_P,:3:],ax[i]) , P_clouds[IDs_P,4+int(2*i+1)]-Soloff(P_clouds[IDs_P,:3:],ay[i]) ]).T 
                            #plt.figure()    
                            #plt.plot(d[:,0],d[:,1],'o',c='red')
                            #plt.plot(np.mean(d[:,0]),np.mean(d[:,1]),'o',c='green')
                            #plt.show()
                            #sys.exit()
                            
                            if len(IDs_P) > 0 and len(IDs_marker) > 0:
                                d = np.vstack([ P_clouds[IDs_P,4+int(2*i)]-Soloff(P_clouds[IDs_P,:3:],ax[i]) , P_clouds[IDs_P,4+int(2*i+1)]-Soloff(P_clouds[IDs_P,:3:],ay[i]) ]).T 
                                # correct marker by maximum peak of disparity
                                x_corr = np.append(x_corr , x[IDs_marker] - np.mean(d[:,0]))
                                y_corr = np.append(y_corr , y[IDs_marker] - np.mean(d[:,1])) 
                                XYZ_corr = np.append(XYZ_corr , XYZ[IDs_marker] , axis=0)
                            elif IDs_P.size == 0 and IDs_marker.size>0:
                                x_corr = np.append(x_corr , x[IDs_marker])
                                y_corr = np.append(y_corr , y[IDs_marker]) 
                                XYZ_corr = np.append(XYZ_corr , XYZ[IDs_marker] , axis=0)
                            else:
                                print('Exit - no marker points in volume')
                                sys.exit()
                #sys.exit()
                # refine calibration parameters
                markerCorr = np.append(np.vstack([x_corr,y_corr]).T, XYZ_corr, axis=1)
                sx , sy = Calibration(markerCorr, [ax[i],ay[i]])
                ax[i] , ay[i] = sx , sy
                # save calibration parameters for the current run
                np.savetxt(params.calibration_output.format(cam=c,xy='x')+'_subVol'+str(v)+'_run'+str(run)+'.txt',sx)
                np.savetxt(params.calibration_output.format(cam=c,xy='y')+'_subVol'+str(v)+'_run'+str(run)+'.txt',sy)  
                params.calibration_path = "../code/preProcessing/9_VSC/parameter/c{cam}/soloff_c{cam}{xy}_subVol"+str(v)+"_run"+str(run)+".txt" 
                # print current calibration result
                d = np.vstack([ markerCorr[:,0]-Soloff(markerCorr[:,2::],ax[i]) , markerCorr[:,1]-Soloff(markerCorr[:,2::],ay[i]) ]).T 
                print('    c{cam} mean calib error: '.format(cam=c),np.mean(np.linalg.norm(d,axis=1)),' +- ',np.std(np.linalg.norm(d,axis=1)))

                #axis = plt.figure().add_subplot(projection='3d')
                #axis.scatter(markerCorr[:,2],markerCorr[:,3],markerCorr[:,4],c='red')
                #axis.set_xlim(0,1), axis.set_ylim(0,1), axis.set_zlim(0,1)
                #plt.show()
                #plt.figure()    
                #plt.plot(markerCorr[:,0],markerCorr[:,1],'o',c='red')
                #plt.show()
                #sys.exit()
                
            print('')
        print('')
        print('')
if __name__ == "__main__":
    main()