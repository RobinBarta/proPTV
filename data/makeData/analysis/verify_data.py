'''

    This Code calculates parameter of the case.
    
'''


import sys
import numpy as np
import matplotlib.pyplot as plt

# %%

class Parameter():    
    case_name, Zeros = '9000_25', 5
    t_start, t_end, dt = 1, 10, 1
    
# %%


def main():
    # load parameter
    params = Parameter()
    params.data_path = '../output/'+params.case_name+'/origin/origin_{time}.txt'
    
    # interpolation
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)[::params.dt]
    for i in range(len(times)-1):
        xy1 = np.loadtxt(params.data_path.format(time=str(times[i]).zfill(params.Zeros)))[:,-2:]
        xy2 = np.loadtxt(params.data_path.format(time=str(times[i+1]).zfill(params.Zeros)))[:,-2:]
        dxdy = np.linalg.norm(xy1-xy2,axis=1)
        print(np.max(dxdy))
        
        plt.figure()
        plt.hist(dxdy,bins=50)
        plt.show()
        
        XYZ1 = np.loadtxt(params.data_path.format(time=str(times[i]).zfill(params.Zeros)))[:,1:4:]
        XYZ2 = np.loadtxt(params.data_path.format(time=str(times[i+1]).zfill(params.Zeros)))[:,1:4:]
        dxdydz = np.linalg.norm(XYZ1-XYZ2,axis=1)
        print(np.max(dxdydz))
        
        plt.figure()
        plt.hist(dxdydz,bins=50)
        plt.show()
if __name__ == "__main__":
    main()