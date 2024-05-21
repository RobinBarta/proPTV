import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir('../../../data')


# %%

class Parameter():    
    N = 45000
    case_name, string = str(N)+'_new', 'with3'
    
# %%


def main(): 
    # load params
    params = Parameter()
    
    # load data
    a = np.loadtxt(params.case_name+'/analysis/pmp_a'+params.string+'.txt')
    b = np.loadtxt(params.case_name+'/analysis/pmp_b'+params.string+'.txt')
    b[b>1] = 1
    c = np.loadtxt(params.case_name+'/analysis/pmp_c'+params.string+'.txt')
    d = np.loadtxt(params.case_name+'/analysis/pmp_d'+params.string+'.txt') * 500
    hit = np.loadtxt(params.case_name+'/analysis/pmp_hit'+params.string+'.txt')
    
    print('\n')
    print('hit tracks: ' + str(hit/params.N) + ' %')
    print('F = ' + str(np.mean(a)) + ' +- ' + str(np.std(a)))
    print('C = ' + str(np.mean(b)) + ' +- ' + str(np.std(b)))
    print('Cr = ' + str(np.mean(c)) + ' +- ' + str(np.std(c)))
    print('E = ' + str(np.mean(d)) + ' +- ' + str(np.std(d)))
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].hist(a,bins=50)
    ax[0].set_xlabel('F'), ax[0].set_ylabel('counts')
    
    ax[1].hist(b,bins=50)
    ax[1].set_xlabel('C'), ax[1].set_ylabel('counts')
    
    ax[2].hist(c,bins=50)
    ax[2].set_xlabel('Cr'), ax[2].set_ylabel('counts')
    
    ax[3].hist(d,bins=50)
    ax[3].set_xlabel('E'), ax[3].set_ylabel('counts')
    plt.tight_layout(), plt.show()
            
if __name__ == "__main__":
    main()  