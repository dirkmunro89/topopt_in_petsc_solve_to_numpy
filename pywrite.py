import numpy as np

def write(vec):

    rank=vec[0]
    np.save("sens.npy"%rank,vec[1:])
    print('sens')   
    return 0

def read(vec):
    print(vec[0])
    return np.load("sens_%d.npy"%vec[0])

    
