import cv2
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.draw import polygon
from skimage.feature import peak_local_max
        
def Resize(img,a):
    root = tk.Tk()
    h, w = img.shape
    screen_h, screen_w = root.winfo_screenheight(), root.winfo_screenwidth()
    window_h, window_w = screen_h*np.sqrt(0.8), screen_w*np.sqrt(0.8)
    img_resize = cv2.convertScaleAbs(img, alpha=a)
    multiplier = 1
    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
            multiplier = window_h / h
        else:
            multiplier = window_w / w
        img_resize = cv2.convertScaleAbs(cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier), alpha=a)
    return img_resize, multiplier

def click_event1(event, x, y, flags, param):
    global mask_points, multiplier
    mask_points, multiplier = param[0], param[1]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        mask_points.append([int(np.round(y/multiplier)),int(np.round(x/multiplier))])
        print('Corner: x = '+str(x) , 'y = '+str(y))
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()
def CollectMask(img_resize,mask_points,multiplier):
    cv2.imshow('Get Mask', img_resize)
    cv2.setMouseCallback('Get Mask', click_event1, [mask_points, multiplier])
    cv2.waitKey(0)
    return mask_points

def Masking(image,mask,pts):
    rr, cc = polygon(pts[:,0], pts[:,1], mask.shape)
    mask[rr,cc] = 1
    image = np.asarray(image)
    image[np.asarray(mask)==0] = 0
    return image

def Thresholding(image,params):
    image[image<params.threshold] = 0
    thresh = image
    return thresh

def RadialSymmetricCenter(image,params):
    cx , cy = [] , []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)	
        if M["m00"] != 0 and params.minArea < cv2.contourArea(c) < params.maxArea:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            I = image[y,x]
            if I != 0:
                cx.append(x)
                cy.append(y)
    return np.asarray(cx) , np.asarray(cy)


def DeleteArtifacts(cx,cy,img_thresh,artifacts,multiplier):
    img_thresh_resize = cv2.cvtColor(cv2.convertScaleAbs(cv2.resize(img_thresh, (0, 0), fx=multiplier, fy=multiplier), alpha=1),cv2.COLOR_GRAY2RGB)
    for cxi,cyi in zip(cx,cy):
        cv2.circle(img_thresh_resize, (int(np.round(cxi*multiplier)),int(np.round(cyi*multiplier))), 1, (0, 0, 255), 3)
    def click_event2(event, x, y, flags, param):
        global artifacts, multiplier
        artifacts, multiplier = param[0], param[1]
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            artifacts.append([int(np.round(x/multiplier)),int(np.round(y/multiplier))])
            cv2.circle(img_thresh_resize, (x,y), 1, (255, 0, 0), 2)
            print('Deleting Artifact: x = '+str(x) , 'y = '+str(y))
        # checking for right mouse clicks     
        if event==cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
    cv2.namedWindow('Delete Artifacts')
    cv2.setMouseCallback('Delete Artifacts', click_event2, [artifacts,multiplier])
    while True:
        cv2.imshow('Delete Artifacts', img_thresh_resize)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    ID = []
    for a in artifacts:
        min_dist = np.min(np.sqrt((cx-a[0])**2+(cy-a[1])**2))
        ID.append(np.argwhere(np.sqrt((cx-a[0])**2+(cy-a[1])**2)==min_dist)[0][0])
    cx , cy = np.delete(cx,ID), np.delete(cy,ID)
    return cx, cy

def click_event22(event, x, y, flags, param):
    global artifacts_add, multiplier
    artifacts_add, multiplier = param[0], param[1]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        artifacts_add.append([int(np.round(x/multiplier)),int(np.round(y/multiplier))])
        print('Search Artifact: x = '+str(x) , 'y = '+str(y))
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()
def SearchArtifacts(cx,cy,img_thresh,artifacts_add,multiplier):
    h = 15
    cx_add, cy_add = [], []
    img_thresh_resize = cv2.cvtColor(cv2.convertScaleAbs(cv2.resize(img_thresh, (0, 0), fx=multiplier, fy=multiplier), alpha=1),cv2.COLOR_GRAY2RGB)
    for cxi,cyi in zip(cx,cy):
        cv2.circle(img_thresh_resize, (int(np.round(cxi*multiplier)),int(np.round(cyi*multiplier))), 1, (0, 0, 255), 3)
    cv2.imshow('Search Artifacts', img_thresh_resize)
    cv2.setMouseCallback('Search Artifacts', click_event22, [artifacts_add,multiplier])
    cv2.waitKey(0)
    for a in artifacts_add:
        ax, ay = int(np.round(a[0])) , int(np.round(a[1]))
        img_a = np.zeros_like(img_thresh)
        img_a[ay-h:ay+h,ax-h:ax+h] = img_thresh.copy()[ay-h:ay+h,ax-h:ax+h]
        gauss = cv2.GaussianBlur( img_a.astype('uint8') , [5,5] , 1 )
        img_a = np.abs(cv2.addWeighted(img_a.astype('uint8'), 3, gauss, -1, 0))
        a_new = peak_local_max(img_a, num_peaks=1)[0]
        cx_add.append(a_new[1])
        cy_add.append(a_new[0])
    return cx_add, cy_add

def CollectMarkerPoints(img_thresh,cx,cy,marker_lines,multiplier):
    img_thresh_resize = cv2.cvtColor(cv2.convertScaleAbs(cv2.resize(img_thresh, (0, 0), fx=multiplier, fy=multiplier), alpha=1),cv2.COLOR_GRAY2RGB)
    for cxi,cyi in zip(cx,cy):
        cv2.circle(img_thresh_resize, (int(np.round(cxi*multiplier)),int(np.round(cyi*multiplier))), 1, (0, 0, 255), 3)
    def click_event3(event, x, y, flags, param):
        global marker_lines, multiplier
        marker_lines, multiplier = param[0] , param[1]
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            marker_lines.append([x/multiplier,y/multiplier])
            cv2.circle(img_thresh_resize, (x,y), 2, (0, 255, 0), 5)
            print('Collect Corners: x = '+str(x) , 'y = '+str(y))
        # checking for right mouse clicks     
        if event==cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
    cv2.namedWindow('Collect Markers')
    cv2.setMouseCallback('Collect Markers', click_event3, [marker_lines,multiplier])
    while True:
        cv2.imshow('Collect Markers', img_thresh_resize)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    xyl, xyr = np.asarray(marker_lines)[:2,:], np.asarray(marker_lines)[2::,:]
    return xyl, xyr

def Get3DMarkerPosition(xy,i,params):
    dim, d = params.depth[0], params.depth[1][params.plane-1]
    XYZ = np.zeros([len(xy),3])
    if dim == 'x':
        Y0 , Z0 = params.startPoint[1], params.startPoint[2]
        Y , Z = params.spacing*np.arange(0,len(xy))+Y0, params.spacing*i*np.ones(len(xy))+Z0
        XYZ[:,0] , XYZ[:,1] , XYZ[:,2] = d*np.ones(len(xy)) , np.ravel(Y) , np.ravel(Z)
    if dim == 'y':
        X0 , Z0 = params.startPoint[1], params.startPoint[2]
        X , Z = params.spacing*np.arange(0,len(xy))+X0, params.spacing*i*np.ones(len(xy))+Z0
        XYZ[:,0] , XYZ[:,1] , XYZ[:,2] = np.ravel(X) , d*np.ones(len(xy)) , np.ravel(Z)
    if dim == 'z':
        X0 , Y0 = params.startPoint[1], params.startPoint[2]
        X , Y = params.spacing*np.arange(0,len(xy))+X0, params.spacing*i*np.ones(len(xy))+Y0
        XYZ[:,0] , XYZ[:,1] , XYZ[:,2] = np.ravel(X) , np.ravel(Y) , d*np.ones(len(xy))
    return XYZ
def FindMarker(xyl,xyr,centers,img,cx,cy,params):    
    ID_l0, ID_l1 = np.argmin(np.linalg.norm(xyl[0:1]-centers,axis=1)), np.argmin(np.linalg.norm(xyl[1:2]-centers,axis=1))
    linear_model = np.polyfit( [centers[ID_l0,0]+np.random.normal(0,1/1000),centers[ID_l1,0]+np.random.normal(0,1/1000)] , [centers[ID_l0,1]+np.random.normal(0,1/1000),centers[ID_l1,1]+np.random.normal(0,1/1000)] , 1)
    m, n = linear_model[0] , linear_model[1]
    dx, dy = ( cy + cx/m - n ) / ( m + (1/m) ), ( m * ( cy + cx/m - n ) / ( m + 1/m ) ) + n
    IDs_l = np.argwhere(np.sqrt((cx-dx)**2+(cy-dy)**2)<params.distance_line)[:,0]
    xy_l = np.array(sorted(centers[IDs_l],key=lambda e:e[1]))[::-1]
    # find marker corners in the right
    ID_r0, ID_r1 = np.argmin(np.linalg.norm(xyr[0:1]-centers,axis=1)), np.argmin(np.linalg.norm(xyr[1:2]-centers,axis=1))
    linear_model = np.polyfit( [centers[ID_r0,0]+np.random.normal(0,1/1000),centers[ID_r1,0]+np.random.normal(0,1/1000)] , [centers[ID_r0,1]+np.random.normal(0,1/1000),centers[ID_r1,1]+np.random.normal(0,1/1000)] , 1)
    m, n = linear_model[0] , linear_model[1]
    dx, dy = ( cy + cx/m - n ) / ( m + (1/m) ), ( m * ( cy + cx/m - n ) / ( m + 1/m ) ) + n
    IDs_r = np.argwhere(np.sqrt((cx-dx)**2+(cy-dy)**2)<params.distance_line)[:,0]
    xy_r = np.array(sorted(centers[IDs_r],key=lambda e:e[1]))[::-1]
    # find markers
    plt.figure()
    i, marker_points = 0, np.empty([0,5])
    for l,r in tqdm(zip(xy_l,xy_r), desc=' find markers per line', position=0, leave=True, delay=0.5):
        linear_model = np.polyfit( [l[0],r[0]] , [l[1],r[1]] , 1)
        m, n = linear_model[0] , linear_model[1]
        dx, dy = ( cy + cx/m - n ) / ( m + (1/m) ), ( m * ( cy + cx/m - n ) / ( m + 1/m ) ) + n
        IDs = np.argwhere(np.sqrt((cx-dx)**2+(cy-dy)**2)<params.distance_line)[:,0]
        if l[0]<r[0]:
            xy = np.array(sorted(centers[IDs],key=lambda e:e[0])) 
        else:
            xy = np.array(sorted(centers[IDs],key=lambda e:e[0]))[::-1]
        XYZ = Get3DMarkerPosition(xy,i,params)
        marker_points = np.append(marker_points,np.append(xy,XYZ,axis=1),axis=0)
        # plot line fit
        plt.imshow(img,cmap='gray',vmax=np.mean(img))
        plt.plot(xy[:,0],xy[:,1],'o',c='red')
        plt.plot([l[0],r[0]] ,m*np.asarray([l[0],r[0]])+n,'-',c='blue')
        plt.plot(cx,cy,'.',c='green')
        i+=1
    plt.show()
    return marker_points
