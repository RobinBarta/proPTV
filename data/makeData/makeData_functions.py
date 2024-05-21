import sys, random, cv2, h5py
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
#import imageio.v2 as imageio
import imageio
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import TwoSlopeNorm

'''
    This scirpt contains functions used to generate camera images from DNS output
'''

def Gauss(x,y,I,xmean,ymean,sigma):
    # Gauss function to blur a particle on an image
    X = (x-xmean) / sigma
    Y = (y-ymean) / sigma
    return I * np.exp( -0.5*(X**2+Y**2) ) / (2*np.pi*sigma**2)

def Soloff(X,Y,Z,a):
    # Soloff calibration function: estimates camera image position from 3D particle position
    return ( a[0] 
                + X * ( X*(a[9]*X+a[11]*Y+a[14]*Z+a[4]) + a[13]*Y*Z + a[6]*Y + a[7]*Z + a[1] ) 
                + Y * ( Y*(a[12]*X+a[10]*Y+a[15]*Z+a[5]) + a[8]*Z + a[2] ) 
                + Z * ( Z*(a[17]*X+a[18]*Y+a[16]) + a[3] ) ) 

def LoadNETCDF(t,List,params):
    # load the nc particle file per time step and write out partilce fields and informations
    data = nc.Dataset(params.path_input.format(time=str(int(t)).zfill(8)))
    Gr, Pr,  = data.variables['gr'][0], data.variables['pr'][0]
    ti, x, y, z = data.variables['tprobl'][0], data['particles'][List,2], data['particles'][List,1], data['particles'][List,0]
    u, v, w, T, p = data['particles'][List,5], data['particles'][List,4], data['particles'][List,3], data['particles'][List,6], data['particles'][List,7]
    return ti, x, y, z, u, v, w, T, p, Gr*Pr, Pr

def GenerateImage(particles,params):
    img = np.zeros((params.y_res,params.x_res)).astype('uint'+str(params.bit))
    # go over each particle and create the blob and put it on the image
    for p in particles:
        px, py, I = p
        x = np.array([ np.rint(px)-2, np.rint(px)-1,  np.rint(px),   np.rint(px)+1,  np.rint(px)+2,
                       np.rint(px)-2, np.rint(px)-1,  np.rint(px),   np.rint(px)+1,  np.rint(px)+2, 
                       np.rint(px)-2, np.rint(px)-1,  np.rint(px),   np.rint(px)+1,  np.rint(px)+2, 
                       np.rint(px)-2, np.rint(px)-1,  np.rint(px),   np.rint(px)+1,  np.rint(px)+2,
                       np.rint(px)-2, np.rint(px)-1,  np.rint(px),   np.rint(px)+1,  np.rint(px)+2,]).astype('uint'+str(params.bit))
        y = np.array([ np.rint(py),   np.rint(py),   np.rint(py),   np.rint(py),   np.rint(py),  
                       np.rint(py)+1, np.rint(py)+1, np.rint(py)+1, np.rint(py)+1, np.rint(py)+1,
                       np.rint(py)+2, np.rint(py)+2, np.rint(py)+2, np.rint(py)+2, np.rint(py)+2,
                       np.rint(py)-1, np.rint(py)-1, np.rint(py)-1, np.rint(py)-1, np.rint(py)-1,
                       np.rint(py)-2, np.rint(py)-2, np.rint(py)-2, np.rint(py)-2, np.rint(py)-2]).astype('uint'+str(params.bit))
        I = p[2]*np.random.normal(1,params.dI/100)
        # calculate the blob intenity distribution, the sum over I_blob is equal to I
        I_blob = np.rint(Gauss(x,y,I,px,py,params.sigma)).astype('uint'+str(params.bit))
        #img += I_blob 
        for i in range(len(x)):
            img[y[i],x[i]] += I_blob[i]
    # add noise
    noise = np.random.uniform(0,np.sqrt(np.mean(img.copy().astype(float)[img>0]**2)/params.SNR), size=(params.y_res,params.x_res)).astype('uint'+str(params.bit))
    img = img + noise 
    return img

def GenerateMarker(cam,ax,ay,params):
    # generate marker images for the calibration
    data = np.empty([0,5])
    for plane,d in enumerate(np.linspace(0,1,params.N_planes)):
        # create output image
        img = np.zeros([params.y_res,params.x_res]).astype('uint8')
        # get 3D world coordinates of the marker plane
        X = d*np.ones([params.N_marker,params.N_marker])
        Y, Z = np.meshgrid(np.linspace(0,1,params.N_marker),np.linspace(0,1,params.N_marker))
        X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
        # estimate 2D pixel coordinates of the marker points
        x, y = Soloff(X,Y,Z,ax), Soloff(X,Y,Z,ay)
        # add x,y to the image and save it
        img[np.rint(y).astype('uint'+str(params.bit)),np.rint(x).astype('uint'+str(params.bit))] = 250
        img = cv2.GaussianBlur( img.astype('uint8') , [31,31] , 1 )
        img = cv2.addWeighted( img, 20, img, 0, 0)  
        cv2.imwrite(params.path_output_marker.format(cam=cam,plane=str(plane+1),time=str(1).zfill(params.zFill),Number=int(params.N_particles),times=int(params.freq)),img)
        # create output data
        data = np.append(data,np.vstack([np.round(x,5),np.round(y,5),X,Y,Z]).T,axis=0)
    np.savetxt(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/calibration_images/markers_c'+str(cam)+'.txt',data,header='x,y,X,Y,Z')
    return 0

def Animation(c,fps,Frames,params):
    # write out mp4
    height, width, layers = cv2.imread(params.path_output_img.format(cam=c,time=str(0).zfill(params.zFill),Number=int(params.N_particles),times=int(params.freq))).shape
    video = cv2.VideoWriter(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+"/particle_images/c{cam}.mp4".format(cam=c),cv2.VideoWriter_fourcc(*'XVID'),fps,(width,height))
    for i,t in enumerate(tqdm(Frames, leave=True, position=0, delay=0.1,desc='Creating Movie: ')):
        img = cv2.imread(params.path_output_img.format(cam=c,time=str(i).zfill(params.zFill),Number=int(params.N_particles),times=int(params.freq)))
        img = cv2.addWeighted( img, 5, img, 0, 0)  
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    return 0

def BuildTracks(Frames,params):
    # create empty track arrays for each particle
    X, Y, Z, VX, VY, VZ = np.empty([params.N_particles,0]), np.empty([params.N_particles,0]), np.empty([params.N_particles,0]), np.empty([params.N_particles,0]), np.empty([params.N_particles,0]), np.empty([params.N_particles,0])
    # load particle information at each time step, build each track
    for i,t in enumerate(tqdm(Frames, leave=True, position=0, delay=0.1,desc='Creating Tracks: ')):
        data = np.loadtxt(params.path_output_origin.format(time=str(i).zfill(params.zFill),Number=int(params.N_particles),times=int(params.freq)),skiprows=1)[:,1::]
        x, y, z = data[:,0:1], data[:,1:2], data[:,2:3]
        vx, vy, vz = data[:,3:4], data[:,5:6], data[:,6:7]
        X, Y, Z, VX, VY, VZ = np.append(X,x,axis=1), np.append(Y,y,axis=1), np.append(Z,z,axis=1), np.append(VX,vx,axis=1), np.append(VY,vy,axis=1), np.append(VZ,vz,axis=1)
    # create output file    
    tracks = h5py.File(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/tracks_origin.hdf5', 'w')
    # load particle information at each time step, build each track and store in output file    
    for i in tqdm(range(params.N_particles), leave=True, position=0 , delay=0.1,desc='Save tracks: '):
        time, pos, vel = np.linspace(0,len(Frames)-1,len(Frames)).reshape(len(Frames),1), np.vstack([X[i],Y[i],Z[i]]).T , np.vstack([VX[i],VY[i],VZ[i]]).T
        datas = np.hstack([time, pos, vel])
        tracks.create_dataset(str(i), datas.shape, dtype='float64', data=datas)
    tracks.close()
    # load tracks
    data = h5py.File(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/tracks_origin.hdf5', 'r')
    tracks = [data[key][:] for key in tqdm(list(data.keys()),leave=True,position=0,desc='Loading Tracks: ',delay=0.5)]
    data.close()
    # plot tracks
    plt.figure(figsize=(10,10))
    axis = plt.axes(projection ='3d')
    axis.set_xlabel('X'), axis.set_ylabel('Y'), axis.set_zlabel('Z')
    axis.set_xlim(0,1), axis.set_ylim(0,1), axis.set_xlim(0,1)
    cmap = plt.get_cmap('seismic')
    norm = plt.Normalize(-0.1,0.1) # TwoSlopeNorm(0)
    for track in tqdm(tracks,leave=True,position=0,desc='Plot Tracks: ',delay=0.5):
        xyz = track[:,1:4:]
        w = track[:,-1]
        points = np.array([xyz[:,0],xyz[:,1],xyz[:,2]]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs,cmap=cmap,norm=norm,linewidths=0.5,alpha=1)
        lc.set_array(w)
        axis.add_collection3d(lc)
    cbar = plt.colorbar(lc)
    cbar.set_label(r'$\vec{u}\cdot\vec{e}_z$')
    axis.view_init(elev=27, azim=38, roll=0)
    plt.savefig(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/tracks_origin.tif',dpi=300)
    plt.show()
    plt.close('all')
    return 0

def CameraPlacement(params):
    x0 , x1 , Nx = 0 , 1 , 10
    y0 , y1 , Ny = 0 , 1 , 10
    z0 , z1 , Nz = 0 , 1 , 10
    c0 , c1 , c2 , c3 = [1.4,0,1] , [1.4,0,0] , [1.4,1,1] , [1.4,1,0]
    LW = 1

    X , Y , Z = np.meshgrid(np.linspace(x0,x1,Nx),np.linspace(y0,y1,Ny),np.linspace(z0,z1,Nz))
    XYZ = np.vstack([np.ravel(X),np.ravel(Y),np.ravel(Z)]).T
    fig = plt.figure(figsize=(10,10))
    axis = plt.axes(projection ='3d')
    axis.scatter(X,Y,Z,color='black',s=0.2)
    axis.plot([1,0],[0,0],[0,0],c='black',lw=LW,zorder=10)
    axis.plot([0,0],[1,0],[0,0],c='black',lw=LW,zorder=10)
    axis.plot([1,1],[1,0],[0,0],c='black',lw=LW,zorder=10)
    axis.plot([1,0],[1,1],[0,0],c='black',lw=LW,zorder=10)
    axis.plot([0,0],[0,0],[1,0],c='black',lw=LW,zorder=0)
    axis.plot([1,1],[0,0],[1,0],c='black',lw=LW,zorder=10)
    axis.plot([0,0],[1,1],[1,0],c='black',lw=LW,zorder=10)
    axis.plot([1,1],[1,1],[1,0],c='black',lw=LW,zorder=10)
    axis.plot([1,0],[0,0],[1,1],c='black',lw=LW,zorder=10)
    axis.plot([0,0],[1,0],[1,1],c='black',lw=LW,zorder=10)
    axis.plot([1,1],[1,0],[1,1],c='black',lw=LW,zorder=10)
    axis.plot([1,0],[1,1],[1,1],c='black',lw=LW,zorder=10)
    axis.scatter(c0[0],c0[1],c0[2],color='red',label='c0')
    axis.scatter(c1[0],c1[1],c1[2],color='orange',label='c1')
    axis.scatter(c2[0],c2[1],c2[2],color='green',label='c2')
    axis.scatter(c3[0],c3[1],c3[2],color='blue',label='c3')
    axis.plot([c0[0],0.5],[c0[1],0.5],[c0[2],0.5],color='red')
    axis.plot([c1[0],0.5],[c1[1],0.5],[c1[2],0.5],color='orange')
    axis.plot([c2[0],0.5],[c2[1],0.5],[c2[2],0.5],color='green')
    axis.plot([c3[0],0.5],[c3[1],0.5],[c3[2],0.5],color='blue')
    axis.set_xlim(0,1.4)
    axis.set_ylim(0,1)
    axis.set_zlim(0,1)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    plt.legend()
    plt.savefig(params.path_output.format(Number=int(params.N_particles),times=int(params.freq))+'/camera_placement.tif',dpi=300)
    plt.show()
    plt.close('all')
    return 0
