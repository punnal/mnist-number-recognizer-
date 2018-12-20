# Importing important libraries
import csv
import numpy as np
import statistics as stats
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#Functions of different algorithms

# Logistic_regression
def log_reg(Xtrain, ytrain, Xval, yval):

	# Training on Data
	reg = linear_model.LogisticRegression()
	reg.fit(Xtrain, ytrain)

	# Predicting
	ypred = reg.predict(Xval)

	#calculating score
	return [reg.score(Xtrain,ytrain), reg.score(Xval,yval)]

# SVM
def s_v_m(Xtrain, ytrain, Xval, yval):

	# Training on Data
	clf = SVC()
	clf.fit(Xtrain, ytrain)

	# Predicting
	ypred = clf.predict(Xval)

	#calculating score
	return [clf.score(Xtrain ,ytrain), clf.score(Xval,yval)]

# Neural Network
def n_n(Xtrain, ytrain, Xval, yval):

	# Training on Data
	nn = MLPClassifier(hidden_layer_sizes=(15,10))
	nn.fit(Xtrain, ytrain)

	# Predicting
	ypred = nn.predict(Xval)

	#calculating score
	return [nn.score(Xtrain,ytrain), nn.score(Xval,yval)]


# Declearing lists to store data
data_list = [] 
data_label = []

# Reading csv File
with open("train.csv") as csvfile:
	print("Loading Data...")
	readCsv = csv.reader(csvfile, delimiter=',')
	# Storing Data
	next(readCsv)
	for rows in readCsv:
		data_label.append(float(rows[0]))
		data_list.append(list(map(float, rows[1:])))		
print("Done")
# Declaring necessary Variables
n = len(data_label)
trainDataLen = int(0.8*n)

#Declaring X and y
Xtrain = np.array(data_list[0: trainDataLen])
ytrain = np.array(data_label[0: trainDataLen])
Xval = np.array(data_label[trainDataLen:])
yval = np.array(data_label[trainDataLen:])

# Logistic regressions
print("Training using Logistic regression")
print ("score on logistic regression = %s" %log_reg(Xtrain, ytrain, Xval, yval))

# SVM
print("Training using SVM")
print ("score on SVM = %s" %s_v_m(Xtrain, ytrain, Xval, yval))

# Neural Network
print("Training using NN")
print ("score on Neural Network = %s" %n_n(Xtrain, ytrain, Xval, yval))