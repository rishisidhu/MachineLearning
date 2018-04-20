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

'''
SIGMOID computes sigmoid of a number
'''
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
'''
PREDICT Predict the label of an input given a trained neural network
   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
 trained weights of a neural network (Theta1, Theta2)
'''		
def predict(Theta1, Theta2, X):
  
	# Useful values
	(r,c) = X.shape
	(num_labels,num_hidden) = Theta2.shape
	
	# You need to return the following variables correctly 
	p = np.zeros((r, 1))
	X = np.c_[np.ones(r), X] # Add a column of ones to x
	
	# ====================== YOUR CODE HERE ======================
	# Instructions: Complete the following code to make predictions using
	#               your learned neural network. You should set p to a 
	#               vector containing labels between 1 to num_labels.
	#
	# Hint: The max function might come in useful. In particular, the max
	#       function can also return the index of the max element, for more
	#       information see 'help max'. If your examples are in rows, then, you
	#       can use max(A, [], 2) to obtain the max for each row.
	#

	z_2 = np.matmul(X,np.transpose(Theta1))
	a_2 = sigmoid(z_2)
	(a_m, a_n) = a_2.shape
	a_2 = np.c_[np.ones(a_m), a_2] #Adding A column of 1's
	z_3 = np.matmul(a_2, np.transpose(Theta2))
	h_theta=sigmoid(z_3)
	p = np.argmax(h_theta, axis=1)

	return p
	# =========================================================================


	
		

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

#Extract Data
print '\nLoading Input Data and Labels ...\n'
data = scipy.io.loadmat('ex3data1.mat')
X = np.array(data['X'])
Y = np.array(data['y'])

print "Images Dimension:", X.shape
print "Labels Dimension:", Y.shape

#Plotting The Image
show_images(0)


#Setup the parameters you will use for this exercise
input_layer_size  	= 400  	# 20x20 Input Images of Digits
hidden_layer_size 	= 25;   # 25 hidden units
num_labels 			= 10;   # 10 labels, from 1 to 10   
							# (note that we have mapped "0" to label 10)

(m,n) = X.shape

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print '\nLoading Saved Neural Network Parameters ...\n'

# Load the weights into variables Theta1 and Theta2
weights = scipy.io.loadmat('ex3weights.mat')
Theta1 = np.array(weights['Theta1'])
Theta2 = np.array(weights['Theta2'])
print 'Size of Theta1:', Theta1.shape
print 'Size of Theta2:', Theta2.shape

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
y_new = Y%10

accuracy=0
for i in range(len(y_new)):
	if(y_new[i] == (pred[i]+1)%10):
		accuracy+=1
print "\nAccuracy of the System:",100*float(accuracy)/len(pred),'%'
#fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

#fprintf('Program paused. Press enter to continue.\n');
#pause;
