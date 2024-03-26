'''

    This code contains general functioncs for the triangulation of 3D Particles from 2D particle lists.

'''


import joblib, itertools, os, cv2
import numpy as np

from tqdm import tqdm

from functions.soloff import *


def Triangulate3DPoints(ImgPoints,t,ax,ay,params):
    '''
        main code for the triangulation routine
    '''
    # initialise triag output list
    Triag = np.empty([0,4+int(2*len(params.cams))])
    # using iterative processing to process triangulation
    for i in range(params.N_triag):
        # triangulate 3D particles based on image points
        Triag_i = Triangulation(ImgPoints,t,params)
        if len(Triag_i)!=0:
            # merge close particles
            Triag_i = MergeCloseParticles(Triag_i,ax,ay,params)
            # remove 
            Triag_i = RemoveOftenUsedParticles(Triag_i,params)
            Triag = np.append(Triag,Triag_i,axis=0)
            # delete used image points from the particle lists
            ImgPoints = RemoveParticlesFromParticleLists(ImgPoints,Triag_i,params)
    # merge close particles
    Triag = MergeCloseParticles(Triag,ax,ay,params)
    # remove 
    Triag = RemoveOftenUsedParticles(Triag,params)
    # save triangulation of time step t
    np.savetxt(params.triangulation_path+"/Points_{timeString}.txt".format(timeString=str(t).zfill(params.Zeros)), Triag, header='X,Y,Z,err,'+''.join([val for val in ['cx'+str(i)+',cy'+str(i)+',' for i in np.arange(len(params.cams))]]))
    return np.unique(Triag,axis=0), ImgPoints

def Triangulation(ImgPoints,t,params):
    '''
        calculate a 3D position for all permutations of the camera views
    '''
    # define output list containing the triangulation
    Triag = []
    currentCams = np.asarray([params.cams])
    # get permutation of camera order
    for camPermute in params.startCamForPermute:
        permutedCams = np.append(np.array([camPermute]),np.asarray(params.cams)[np.argwhere(currentCams[0]!=camPermute)[:,0]])
        currentCams = np.append(currentCams,np.array([permutedCams]),axis=0)
    # define original camera orientation to be currentCams[0]
    ax_current = np.asarray([np.loadtxt(params.calibration_path.format(cam=cam,xy="x"),delimiter=',') for cam in currentCams[0]])
    ay_current = np.asarray([np.loadtxt(params.calibration_path.format(cam=cam,xy="y"),delimiter=',') for cam in currentCams[0]])
    for cams in currentCams:
        # get calibration files and the particle lists
        aX = np.asarray([np.loadtxt(params.calibration_path.format(cam=cam,xy="x"),delimiter=',') for cam in cams])
        aY = np.asarray([np.loadtxt(params.calibration_path.format(cam=cam,xy="y"),delimiter=',') for cam in cams])
        imgs_cams = [cv2.imread(params.case_path+'input/raw_images/c{c}/c{c}_{time}.tif'.format(c=cam,time=str(t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) for cam in cams]
        ImgPoints_cams = list(np.asarray(ImgPoints,dtype=object)[ [np.argwhere(cam==params.cams)[0][0] for cam in cams] ])
        Triag += joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(GetTriangulationCandidates)(point,ImgPoints_cams,imgs_cams,aX,aY,cams,currentCams,params) for point in tqdm(ImgPoints_cams[0], desc='   triangulate '+str(cams)+': ', position=0 , leave=True, delay=0.5 ))
    return np.array([P for P in Triag if P!=[]])

def GetTriangulationCandidates(point,ImgPoints_cams,img,aX,aY,cams,currentCams,params):
    '''
        # Calculates all possible candidates of image points for the triangulation based on epi polar geometry
    '''
    # select first point
    p1 = point[:2:]
    # get unused cams
    restcams = np.arange(1,len(params.cams))
    ''' Step 1 - get epi lines '''
    # calculate points that project onto p1 for camera with index 0
    p1_same = GetNCorrespondingImgPoints(p1,5,params.depthaxis[cams[0]], aX[0], aY[0], params)
    # calculate epi line onto the next camera , e.g. the camera with index 1
    epiline = np.asarray([GetEpiLine(p1_same, aX[c], aY[c]) for i,c in enumerate(restcams)])
    m_epi, n_epi = epiline[:,0], epiline[:,1]
    ''' Step 2 - find all points near epi line in perpendicular distance epsD '''
    remainingPoints = [RemainingPointsNearEpiLine(ImgPoints_cams[c], m_epi[i], n_epi[i], p1_same, aX[c], aY[c], params.epsD) for i,c in enumerate(restcams)]
    ''' Step 3 - triangulate a test particle for each remainingPoints and p1 and collect all possible cam points '''
    camPs = [ np.array([p1,np.array([np.nan,np.nan])]) ] + [np.array([[np.nan,np.nan]]) for i,c in enumerate(restcams)]
    for i,c in enumerate(restcams):
        testTriag = np.asarray([NewtonSoloff_Triangulation([p1,p2], [aX[0],aX[c]] , [aY[0],aY[c]], params) for p2 in remainingPoints[i]],dtype=object)
        surviverID = np.asarray([ID for ID in np.arange(len(testTriag)) if all(params.Vmin<testTriag[ID,0]) and all(testTriag[ID,0]<params.Vmax) and np.asarray([ele<params.eps for ele in testTriag[ID,1]]).all() and IntensityCheck(img,testTriag[ID,0],aX,aY,params)])
        if len(surviverID) > 0:
            testTriag = testTriag[surviverID]
            remainingPoints[i] = remainingPoints[i][surviverID]
            for n in range(len(testTriag)):
                camPs[c] = np.append(camPs[c],np.array([remainingPoints[i][n]]),axis=0)
                for ci in restcams[restcams!=c]:
                     for remPC in RemainingPointsInsideCircle(ImgPoints_cams[ci], [F(testTriag[n][0],aX[ci]) , F(testTriag[n][0],aY[ci])], params.epsC):
                         camPs[ci] = np.append(camPs[ci],np.array([remPC]),axis=0)
    if len(camPs)>0:
        camPs_u = [np.unique(setP,axis=0) for setP in camPs]
        ''' Step 4 - calculate all possible pairs'''
        pairs = np.asarray([combination for combination in list(itertools.product(*camPs_u)) if len(np.argwhere(np.isnan(combination)[:,0]==False)[:,0]) >= params.activeMatches_triag])
        # swap cam orientation and triangulate pairs
        camOrientation = [ np.argwhere(currentCams[0][i]==cams)[0][0] for i in np.arange(len(cams)) ]
        if len(pairs)>0:
            pairs = pairs[:,camOrientation]
            aX, aY = aX[camOrientation], aY[camOrientation]
            img = list(np.asarray(img)[camOrientation])
            ''' Step 5 - triangulate pairs '''
            triag = np.asarray([NewtonSoloff_Triangulation(setP, aX, aY, params) for setP in pairs],dtype=object)
            surviverID = np.asarray([ID for ID in np.arange(len(triag)) if np.asarray([ele<params.eps for ele in triag[ID,1]]).all() and IntensityCheck(img,triag[ID,0],aX,aY,params) ])
            if len(surviverID)>0:
                triag = triag[surviverID]
                pairs = pairs[surviverID]
                ERR = np.asarray([np.mean(ele) for ele in triag[:,1]])
                ''' Step 6 - search best pair'''
                ID_final = np.argmin(ERR)
                triag_final = triag[ID_final]
                camP_final = pairs[ID_final]
                return [triag_final[0][0],triag_final[0][1],triag_final[0][2],ERR[ID_final]] + list(np.ravel(camP_final))              
    return [] 


def IntensityCheck(img,P,aX,aY,params):
    return (np.array([params.Imin])<[img[i][int(np.rint(F(P,aY[i]))),int(np.rint(F(P,aX[i])))] for i in range(len(params.cams))]).all()  

def DistanceToFunction(p, coeffs, x):
    def d(x, p, coeffs):
        x_curve, y_curve = x, np.polyval(coeffs, x)
        return np.sqrt((x_curve - p[0])**2 + (y_curve - p[1])**2)
    res = minimize(lambda x: d(x, p, coeffs), x0=p[0], bounds=[[np.min(x),np.max(x)]])
    return res.fun, res.x[0]

def Polynomial(x, coeffs):
    return np.polyval(coeffs, x)

def GetNCorrespondingImgPoints(p, N, depthaxis, ax, ay, params):
    '''
        Calculate N 3D Points that project onto the same image point. The 3D Points are used to 
        calculate the epipolarlines in different cams by reprojection.
    '''  
    P = np.zeros([N,3])
    P[:,0] = (params.Vmax[0]+params.Vmin[0])/2
    P[:,1] = (params.Vmax[1]+params.Vmin[1])/2
    P[:,2] = (params.Vmax[2]+params.Vmin[2])/2
    P[:,depthaxis] = np.linspace(params.Vmin[depthaxis]-0.25*(np.abs(np.mean([params.Vmin[depthaxis],params.Vmax[depthaxis]]))),params.Vmax[depthaxis]+0.25*(np.abs(np.mean([params.Vmin[depthaxis],params.Vmax[depthaxis]]))),N)
    #P[:,depthaxis] = np.linspace(params.Vmin[depthaxis]-(np.abs(params.Vmin[depthaxis])*0.0),params.Vmax[depthaxis]+(np.abs(params.Vmax[depthaxis])*0.0),N)
    for n in range(N):
        for i in range(3):
            P[n,:] += np.linalg.lstsq(Jacobian_Soloff(P[n,:],[ax],[ay]),-np.array([F(P[n,:],ax)-p[0], F(P[n,:],ay)-p[1]]),rcond=None)[0] 
        if (P[n,depthaxis]>(params.Vmax[depthaxis]+params.Vmin[depthaxis])/2) and (P[n,depthaxis]<params.Vmax[depthaxis]) and (n==N-1):
            P[n,depthaxis] = params.Vmax[depthaxis] + (np.abs(params.Vmax[depthaxis])*0.1)
            for i in range(2):
                P[n,:] += np.linalg.lstsq(Jacobian_Soloff(P[n,:],[ax],[ay]),-np.array([F(P[n,:],ax)-p[0], F(P[n,:],ay)-p[1]]),rcond=None)[0] 
        elif (P[n,depthaxis]<(params.Vmax[depthaxis]+params.Vmin[depthaxis])/2) and (P[n,depthaxis]>params.Vmin[depthaxis]) and (n==0):
            P[n,depthaxis] = params.Vmin[depthaxis] - (np.abs(params.Vmin[depthaxis])*0.1)
            for i in range(2):
                P[n,:] += np.linalg.lstsq(Jacobian_Soloff(P[n,:],[ax],[ay]),-np.array([F(P[n,:],ax)-p[0], F(P[n,:],ay)-p[1]]),rcond=None)[0] 
    return P 

def Get2CorrespondingImgPoints(p, ax, ay, params):
    '''
        Calculate 2 3D Points that project onto the same image point. The 3D Points are used to 
        calculate the epipolarlines in different cams by reprojection.
    '''  
    P1 = np.asarray(params.Vmin) + 0.25*(np.asarray(params.Vmax)-np.asarray(params.Vmin))
    P2 = np.asarray(params.Vmax) - 0.25*(np.asarray(params.Vmax)-np.asarray(params.Vmin))
    for i in range(6):
        P1 += np.linalg.lstsq(Jacobian_Soloff(P1,[ax],[ay]),-np.array([F(P1,ax)-p[0], F(P1,ay)-p[1]]),rcond=None)[0]
        P2 += np.linalg.lstsq(Jacobian_Soloff(P2,[ax],[ay]),-np.array([F(P2,ax)-p[0], F(P2,ay)-p[1]]),rcond=None)[0]   
    return [P1,P2] 

def GetEpiLine(P_epi, ax, ay): 
    '''
        Calculates the Epiline slope and offset.
    '''
    linear_model = np.polyfit( [F(P,ax) for P in P_epi] , [F(P,ay) for P in P_epi] , 1)
    return linear_model[0] , linear_model[1] #m , n = ((y2-y1) / (x2-x1)) , (y1 - ((y2-y1) / (x2-x1))*x1)

def RemainingPointsNearEpiLine(ImgPoints, m, n, p1_same, ax, ay, epsD): 
    '''
        All image points within the distance epsD around the epiline.
    '''  
    Fp = np.asarray([[F(pi,ax),F(pi,ay)] for pi in p1_same])
    dx = ( ImgPoints[:,1] + ImgPoints[:,0]/m - n ) / ( m + (1/m) )
    dy = ( m * ( ImgPoints[:,1] + ImgPoints[:,0]/m - n ) / ( m + 1/m ) ) + n
    return ImgPoints[ np.argwhere( (np.sqrt( (ImgPoints[:,0] - dx)**2 + (ImgPoints[:,1] - dy)**2) < epsD) &
                                   (ImgPoints[:,1]>np.min(Fp[:,1])-epsD) & (ImgPoints[:,1]<np.max(Fp[:,1])+epsD) &
                                   (ImgPoints[:,0]>np.min(Fp[:,0])-epsD) & (ImgPoints[:,0]<np.max(Fp[:,0])+epsD) )[:,0] ] 

def EpiIntersectionPoint(m1,m2,n1,n2):  
    '''
        Calculates the intersection point of two epipolar lines
    ''' 
    xs = (n2-n1)/(m1-m2)
    ys = m1*(n2-n1)/(m1-m2) + n1
    return xs , ys

def RemainingPointsInsideCircle(ImgPoints, xy, epsC):
    '''
        Finds all image points within the radius epsC around a given image point.
    '''
    xs , ys = xy[0] , xy[1]
    points = ImgPoints[ np.argwhere( np.sqrt((ImgPoints[:,0] - xs)**2 + (ImgPoints[:,1] - ys)**2) < epsC )[:,0] ]
    if points.size != 0:
        return points[:,:2:]
    return []

def MergeCloseParticles(Triag,ax,ay,params):
    '''
        Post Processing of the Triangulation. Merge close particles together.
    '''
    if params.epsDoubling>0:
        surviver_index = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(MergeParallelLoop)(i,Triag,ax,ay,params) for i in tqdm(range(len(Triag)),desc='   merge close particles: ',position=0,leave=True,delay=0.5))
        surviver_index = np.unique([surviver for surviver in surviver_index if surviver!=[]])
        if surviver_index.size!=0:
            Triag = Triag[surviver_index]
    return Triag
def MergeParallelLoop(i,Triag,ax,ay,params):
    distances = np.linalg.norm(Triag[:,:3:]-Triag[i,:3:],axis=1)
    index = np.argwhere( distances<params.epsDoubling )[:,0]
    doublingPoints = Triag[index]
    camPs = doublingPoints[:,4::]
    activeCams = np.asarray([len(np.argwhere(np.isnan(camP[::2])==False)[:,0]) for camP in camPs])
    costs = doublingPoints[:,3]
    # check for the best triag point, use more cam informations first
    for n in np.linspace(len(params.cams),params.activeMatches_triag,len(params.cams)-params.activeMatches_triag+1):
        nID = np.argwhere(activeCams == n)
        if nID.size != 0:
            bestID = np.argmin(costs[nID[:,0]])
            return int(index[nID[:,0]][bestID])
    return []

def RemoveOftenUsedParticles(Triag,params):
    '''
        Post Processing of the Triangulation. Remove any two or more 3D Point which uses the same image point for the triangulation.
        It keeps only the best Triangulation.
    '''
    for c in tqdm(range(len(params.cams)),desc='   uniqueness filter: ', position=0,leave=True,delay=0.5):
        cTriag = Triag[:,int(4+2*c):int(4+2*(c+1)):]
        TriagErr = Triag[:,3]
        surviverID = []
        for i,p in enumerate(cTriag):
            if np.isnan(p[0])==False:
                IDs = np.argwhere(np.linalg.norm(p-cTriag,axis=1)==0)[:,0]
                surviverID.append(IDs[np.argmin(TriagErr[IDs])])
            else:
                surviverID.append(i)
        if np.unique(surviverID).size!=0:
            Triag = Triag[np.unique(surviverID)]
    return Triag

def RemoveParticlesFromParticleLists(ImgPoints,Triag,params):
    '''
        Remove for Triangulation used cam points from the particle image lists.
    '''
    deleteList = [[np.argwhere( np.linalg.norm(ImgPoints[c]-P,axis=1)==0)[0][0] for P in Triag[:,int(4+2*c):int(4+2*(c+1)):] if np.isnan(P[0])==False] for c in np.arange(len(params.cams))]
    return [np.delete(np.asarray(ImgPoints[i]),np.asarray(deleteList[i]),axis=0) if len(deleteList[i])!=0 else ImgPoints[i] for i in np.arange(len(params.cams))]