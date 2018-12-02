import random
from matplotlib import pyplot as plt

'''
GLOBAL VARIABLES
'''
SAMPLE_SIZE = 50

'''
Function: Straight line
'''
def strLine(xData):
	yData = xData
	return yData

'''
Function: Points on Straight line
'''
def strLinePoints(xData):
	sample_x 	= random.sample(xData, SAMPLE_SIZE)
	remain_x 	= [x for x in xData if x not in sample_x]
	yData 		= [x+2 for x in sample_x] + [x-2 for x in remain_x]
	newx 		= sample_x + remain_x
	return newx, yData
	
x1 = range(100)
y1 = strLine(x1)
x2, y2 = strLinePoints(x1)
plt.plot(x1,y1)
plt.scatter(x2, y2, marker = '*', color = 'r')
plt.title('Straight Line')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()