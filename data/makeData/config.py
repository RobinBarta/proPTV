class Parameter:
    ''' general settings'''
    # input name of case; output name; number of zeros in file names
    case_name, output_name, Zeros  = "27000", "run_back", 5           
    # flag if intial tracks are loaded and path to case where tracks are loaded
    loadOption, load_name, suffix = True, "run1", ""                                                                       
    # cameras + orientation
    cams, depthaxis = [0,1,2,3], [0,0,0,0] # 0=X, 1=Y, 2=Z                                                                  
    # first frame, last frame, initialisation length, delta between frames
    t_start, t_end, t_init, dt = 0, 29, 3, 1
    
    ''' triangulation parameter '''
    # measurment volume [x,y,z]
    Vmin , Vmax = [0,0,0], [1,1,1]   
    # select camera viewing angles 
    startCamForPermute = [3]
    #  number of triangulation loops, minimum number of different cams needed for triag, distance from epipolar line [px], distance from intersection point of epipolar lines [px], maxmium BackProjection error in Newton Soloff, distance to remove doubled tringulation points                                                          
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling, Imin = 3, 3, 1.5, 1.0, 0.5, 0.005, 2950  
    
    ''' initalisation parameter '''  
    # maximal absolute tracking velocity for a track                          
    maxvel, angle = 0.015, 60
    # number of initialisation loops; number of maximal NNs per linking step                                  
    N_init, NN = 3, [3,3,3]
    
    ''' tracking parameter '''
    # active cams for extend                                         
    activeMatches_extend, epsR = 4, 3
    # backtracking and gaptracking option
    backtracking, gaptracking = True, False
