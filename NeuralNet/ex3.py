import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
from random import randrange


'''
Global Parameters
'''
IMSHOW_WAIT = 2 #Seconds

'''
Function to Display Images
'''
def show_images(num_images):
	im_count=0
	while im_count<num_images:
		random_index = randrange(0,len(Y))
		img 		= X[random_index,:]				#img is 1 x 400
		label 		= Y[random_index,0]				#label for img
		np.resize(img,(20,20))						#1 x 400 -> 20 x 20
		img 		= np.reshape(img, (20, 20))
		plt.imshow(img, cmap='gray', interpolation='nearest')
		plt.show(block=False)						#Show Image
		im_count	+=1
		print ("Image Label: ", label%10)			#Print Image Label
		time.sleep(IMSHOW_WAIT)
		plt.close()



#Extract Data
mat = scipy.io.loadmat('ex3data1.mat')
X = np.array(mat['X'])
Y = np.array(mat['y'])

print "Images Dimension:", X.shape
print "Labels Dimension:", Y.shape

#Plotting The Image
show_images(10)
