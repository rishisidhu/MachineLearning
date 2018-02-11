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
	plt.show()
	quit()
	y_data_predicted = np.matmul(X,theta)
	plt.plot(X_data, y_data_predicted, marker='*', linestyle='-', color='b', label='pred')
	plt.legend(loc='lower right')

'''
SIGMOID computes sigmoid of a number
'''
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
	
	
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data 	= pd.read_csv('ex2data1.txt', header =  None, names = ['Exam_1_Score', 'Exam_2_Score', 'Label'])

#Get The Labels
y_data 	= data.iloc[:,2]

#Get the Data
X_data 	= data.iloc[:,0:2]

m 		= len(y_data)						  #Number of training samples
y 		= np.array(y_data)
X 		= np.c_[np.ones(m), np.array(X_data)] # Add a column of ones to x

#Call The Plotting Function
#plotData(X, y)

#Call The Sigmoid Function
#print sigmoid(np.zeros((2,2)))