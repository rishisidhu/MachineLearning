import scipy.io
import numpy as np

mat = scipy.io.loadmat('ex3data1.mat')
X = np.array(mat['X'])
Y = np.array(mat['y'])

print "Images Dimension:", X.shape
print "Labels Dimension:", Y.shape
