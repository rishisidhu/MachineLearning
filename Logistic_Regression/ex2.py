## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.optimize as op
	
'''
PLOTDATA Plots the data points X and y into a new figure 
   PLOTDATA(x,y) plots the data points with + for the positive examples
   and o for the negative examples. X is assumed to be a Mx2 matrix.
'''
def plotData(X, y):
	pos_vals  = np.where(y==1)
	neg_vals  = np.where(y==0)
	pass_rows = X[pos_vals, : ][0]
	fail_rows = X[neg_vals, : ][0]

	# Plot the linear fit
	plt.scatter(pass_rows[:,1], pass_rows[:,2], marker='+',  color='k', label='pass', s = 100) 
	plt.scatter(fail_rows[:,1], fail_rows[:,2], marker='o',  color='y', label='fail', s = 100) 
	plt.legend(loc='lower right')
	plt.xlabel('Exam 1 Scores')
	plt.ylabel('Exam 2 Scores')
	plt.title('Scatter Plot of Pass/Fail Results based on Exam Scores')
	

'''
SIGMOID computes sigmoid of a number
'''
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
	
'''
ComputeCost computes the cost function
'''
def computeCost(theta, X, y):
	#computeCost Compute cost for logistic regression
	#   J = computeCost(X, y, theta) computes the cost of using theta as the
	#	parameter for logistic regression and the gradient of the cost
   #	w.r.t. to the parameters.

	# Initialize some useful values
	m = len(y); # number of training examples

	# You need to return the following variables correctly 
	J = 0;

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	# =========================================================================
	# You should set J to the cost.
	y = y.reshape(m,1)
	z		 	= np.matmul(X,theta) 		# X*theta
	h_theta	  	= sigmoid(z)				# sigmoid
	cost_0		= np.log(1-h_theta)			# Cost For Fail(=0) Term
	cost_1		= np.log(h_theta)			# Cost For Pass(=1) Term
	J			= (1.0/float(m))*((-1*np.matmul(np.transpose(y),cost_1))-np.matmul(np.transpose(1-y),cost_0))
	return J[0]

'''
ComputeGrad computes the Gradient
'''
def computeGrad(theta, X, y):
	#computeGrad Computes gradient for logistic regression
	
	# Initialize some useful values
	(m, n) = X.shape # number of training examples
	
	# You need to return the following variables correctly 
	J = 0;

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	# =========================================================================
	# You should set J to the cost.
	y 			= y.reshape((m,1))
	theta		= theta.reshape((n,1))
	z		 	= np.matmul(X,theta) 		# X*theta
	h_theta	  	= sigmoid(z)				# sigmoid
	grad		= (1.0/float(m))*np.matmul(np.transpose(X),np.subtract(h_theta,y))	
	#print X.shape, theta.shape, y.shape, np.subtract(h_theta,y).shape
	return grad
	
'''
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones
'''
def plotDecisionBoundary(theta, X, y):
	plotData(X,y)
	(m,n) = X.shape
	if(n<=3):
		# Only need 2 points to define a line, so choose two endpoints
		x_min  = int(np.min(X[:,1])-2)
		x_max  = int(np.max(X[:,1])+2)
		plot_x = np.array(range(x_min,x_max))

		# Calculate the decision boundary line
		plot_y = (-1.0/theta[2])*((theta[1]*plot_x) + theta[0])
		#print plot_y
		
		# Plot, and adjust axes for better viewing
		plt.plot(plot_x, plot_y)
		plt.show()
		
		# Legend, specific for the exercise
		#legend('Admitted', 'Not admitted', 'Decision Boundary')
		#axis([30, 100, 30, 100])
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data 	= pd.read_csv('ex2data1.txt', header =  None, names = ['Exam_1_Score', 'Exam_2_Score', 'Label'])

#Get The Labels
y_data 	= data.iloc[:,2]

#Get the Data
X_data 	= data.iloc[:,0:2]
(m, n) 	= X_data.shape

y 		= np.array(y_data)
#  Setup the data matrix appropriately, and add ones for the intercept term
X 		= np.c_[np.ones(m), np.array(X_data)] # Add a column of ones to x

#Call The Plotting Function
#plotData(X, y)
#plt.show()
	
#Call The Sigmoid Function
#print sigmoid(np.zeros((2,2)))

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
#cost= computeCost(initial_theta, X, y)
#print 'Cost is:', cost 
#print 'Gradient Matrix:', grad

#Optimizing Using Built in Scipy Functions
Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X, y),method = 'TNC',jac = computeGrad) #You can also use bfgs for optimization method
optimal_theta = Result.x
print "Optimal Theta: ", Result.x
print "Optimal Cost: ", Result.fun

plotDecisionBoundary(optimal_theta, X, y)