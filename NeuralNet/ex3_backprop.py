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

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_p):
	#nnCostFunction Implements the neural network cost function for a two layer
	#neural network which performs classification
	#NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, X, y, lambda)
	#computes the cost and gradient of the neural network. 

	#The returned parameter grad should be a "unrolled" vector of the
	#partial derivatives of the neural network.
	#

	#Retreive Theta1 and Theta2
	T1 = nn_params[0]
	T2 = nn_params[1]
	
	# Setup some useful variables
	(m,n) = X.shape
			 
	# You need to return the following variables correctly 
	J = 0;
	Theta1_grad = np.zeros(T1.shape)
	Theta2_grad = np.zeros(T2.shape)
	
	#print Theta1_grad.shape
	#print Theta2_grad.shape
	
	# ====================== YOUR CODE HERE ======================
	# Instructions: You should complete the code by working through the
	#               following parts.
	#
	# Part 1: Feedforward the neural network and return the cost in the
	#         variable J. After implementing Part 1, you can verify that your
	#         cost function computation is correct by verifying the cost
	#         computed in ex4.m
	#
	# Part 2: Implement the backpropagation algorithm to compute the gradients
	#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
	#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
	#         Theta2_grad, respectively. After implementing Part 2, you can check
	#         that your implementation is correct by running checkNNGradients
	#
	#         Note: The vector y passed into the function is a vector of labels
	#               containing values from 1..K. You need to map this vector into a 
	#               binary vector of 1's and 0's to be used with the neural network
	#               cost function.
	#
	#         Hint: We recommend implementing backpropagation using a for-loop
	#               over the training examples if you are implementing it for the 
	#               first time.
	#
	# Part 3: Implement regularization with the cost function and gradients.
	#
	#         Hint: You can implement this around the code for
	#               backpropagation. That is, you can compute the gradients for
	#               the regularization separately and then add them to Theta1_grad
	#               and Theta2_grad from Part 2.
	#

	X = np.c_[np.ones(m), X] # Add a column of ones to x
	
	#Layer 2
	z_2 = np.matmul(X,np.transpose(Theta1))
	a_2 = sigmoid(z_2)
	(a_m, a_n) = a_2.shape
	a_2 = np.c_[np.ones(a_m), a_2] #Adding A column of 1's
	
	#Layer 3
	z_3 = np.matmul(a_2, np.transpose(Theta2))
	h_theta=sigmoid(z_3)

	#Cost Components
	cost_0 = np.log(1-h_theta)
	cost_1 = np.log(h_theta)

	#Modified Y Matrix (Ones and Zeros instead of class labels)
	y_new  = np.zeros((m,num_labels))
	#print "Ynew Shape:", y_new.shape
	for i in range(m):
		#Label 0 is at the 10th position. Take Note of that!
		y_new[i,y[i]-1]=1
		
	#Compute Un-regularized Cost J
	J_0 = -1.0*np.multiply(y_new,cost_1)
	J_1 = np.multiply((1-y_new),cost_0)
	J_mat	  = (1.0/float(m))*(J_0-J_1)
	J = np.sum(J_mat)
	
	#Regularized Cost
	T1_shape = T1.shape
	T2_shape = T2.shape
	#Bias Terms Don't Figure in Regularization so 
	#We will ignore the first column of T1 and T2
	denom = 2.0*float(m)
	lambda_ratio = lambda_p/denom
	J = J+lambda_ratio*(np.sum(np.square(T1[:,1:]))+np.sum(np.square(T2[:,1:])))
	
	return J
		

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

#Extract Data
print '\nLoading Input Data and Labels ...\n'
data = scipy.io.loadmat('bpdata1.mat')
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
weights = scipy.io.loadmat('bpweights.mat')
Theta1 = np.array(weights['Theta1'])
Theta2 = np.array(weights['Theta2'])
print 'Size of Theta1:', Theta1.shape
print 'Size of Theta2:', Theta2.shape

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in function nnCostFunction to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  for the fixed debugging parameters.
#
#  I suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print '\nFeedforward Using Neural Network ...\n'

# Weight regularization parameter (we set this to 0 here).
lambda_p = 1.0

nn_params = [Theta1, Theta2]
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_p)

print 'Cost at parameters (loaded from ex4weights):',J,'\n(this value should be about 0.287629 for unregularized and 0.38377 for regularized)\n'




