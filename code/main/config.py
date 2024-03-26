class Parameter:
    ''' general settings'''
    # input name of case; output name; number of zeros in file names
    case_name, output_name, Zeros  = "RBC300", "test", 7           
    # flag if intial tracks are loaded and path to case where tracks are loaded
    loadOption, load_name, suffix = False, "run1", ""                                                                       
    # cameras + orientation
    cams, depthaxis = [0,1,2,3], [1,1,1,1] # 0=X, 1=Y, 2=Z                    
    # first frame, last frame, initialisation length, delta between frames
    t_start, t_end, t_init, dt = 20000, 20020, 3, 1
    
    ''' triangulation parameter '''
    # measurment volume [x,y,z]
    Vmin , Vmax = [0,0,0], [300,300,300]   
    # select camera viewing angles 
    startCamForPermute = []
    #  number of triangulation loops, minimum number of different cams needed for triag, distance from epipolar line [px], distance from intersection point of epipolar lines [px], maxmium BackProjection error in Newton Soloff, distance to remove doubled tringulation points                                                          
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling, Imin = 3, 3, 2, 2, 1, 0, 400 
    
    ''' initalisation parameter '''  
    # maximal absolute tracking velocity for a track                          
    maxvel, angle = 5, 180
    # number of initialisation loops; number of maximal NNs per linking step                                  
    N_init, NN = 2, [3,3,3]
    
    ''' tracking parameter '''
    # active cams for extend                                         
    activeMatches_extend, epsR = 3, 3
    # backtracking and gaptracking option
    backtracking, gaptracking = False, False