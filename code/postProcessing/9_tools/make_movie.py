'''

    This makes a movie of camera images.
    
'''


import os, cv2
import numpy as np

from tqdm import tqdm

os.chdir('../../../data')


# %%

class Units_parameter():    
    case_name, Zeros = 'rbc_300mm_run2', 7
    t_start, t_end, dt = 20000, 20500, 1
    cams = [0]
    
    alpha = 40
    fps = 10
    
# %%

def main(): 
    # load params
    params = Units_parameter()
    params.raw_path = params.case_name+'/input/raw_images/c{cam}/c{cam}_{time}.tif'
    params.video_path = params.case_name+'/analysis/raw_movie.avi'
    
    print('Make movie: ')
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    for cam in params.cams:
        # make avi
        height, width, layers = cv2.imread(params.raw_path.format(cam=cam,time=str(params.t_start).zfill(params.Zeros))).shape
        video = cv2.VideoWriter(params.video_path,cv2.VideoWriter_fourcc(*'XVID'),params.fps,(width,height))
        for t in tqdm(times, desc=' generate movie of cam ' + str(cam), position=0 , leave=True, delay=0.5):
            video.write( cv2.convertScaleAbs(cv2.imread(params.raw_path.format(cam=cam,time=str(t).zfill(params.Zeros))), alpha=params.alpha) )
    cv2.destroyAllWindows()
    video.release()
if __name__ == "__main__":
    main()  