import scipy.io
mat = scipy.io.loadmat('ex3data1.mat')
print type(mat)
print len(mat)
for key,value in mat.items():
	print key
	print '***'
	print value
	print '^^^^^^^^^^^^^^^^^^^^'