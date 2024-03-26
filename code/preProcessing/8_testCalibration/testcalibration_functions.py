import numpy as np
import matplotlib.pyplot as plt

def Soloff(XYZ,a):
    X , Y , Z = XYZ[:,0] , XYZ[:,1] , XYZ[:,2]
    return ( a[0] 
                + X * ( X*(a[9]*X+a[11]*Y+a[14]*Z+a[4]) + a[13]*Y*Z + a[6]*Y + a[7]*Z + a[1] ) 
                + Y * ( Y*(a[12]*X+a[10]*Y+a[15]*Z+a[5]) + a[8]*Z + a[2] ) 
                + Z * ( Z*(a[17]*X+a[18]*Y+a[16]) + a[3] ) ) 