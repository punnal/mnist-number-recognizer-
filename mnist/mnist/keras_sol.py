# Importing important libraries
import csv
import numpy as np
import statistics as stats
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

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
Xtrain = np.array(data_list)
ytrain = np.array(data_label)
print(Xtrain.shape)
print(ytrain.shape)
print(Xtrain.shape)
print(Xtrain.shape)

# Training

model = Sequential([
	Dense(16, input_shape=(784,), activation='relu'),
	Dense(32, activation='relu'),
	Dense(10,  activation='softmax')
	])

model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(Xtrain, ytrain, batch_size=5, epochs=25, shuffle=True, verbose=2, validation_split = 0.2)
