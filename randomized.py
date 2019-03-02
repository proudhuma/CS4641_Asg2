#!/usr/bin/python3.6
import mlrose
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def loadData(filename):
	# load adult data and drop missing information
	data = pd.read_csv(filename, names=['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country','result'],
		engine="python", skipinitialspace=True, na_values=['?'])
	data = data.dropna()

	# encode the labels to numbers
	eData = data.copy()
	for column in eData.columns:
		if eData.dtypes[column] == 'object':
			le = preprocessing.LabelEncoder()
			le.fit(eData[column])
			eData[column] = le.transform(eData[column])
	return eData
	
def prepareData(cross):
	trainset = loadData("adult.data")
	testset = loadData("adult.test")

	# seperate data for cross validation
	trainX, validationX, trainY, validationY = train_test_split(trainset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']], trainset['result'], test_size=cross)
	testX = testset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
	testY = testset["result"]

	scaler = preprocessing.MinMaxScaler()

	trainX_scaled = scaler.fit_transform(trainX)
	validationX_scaled = scaler.transform(validationX)
	testX_scaled = scaler.transform(testX)

	return trainX_scaled, validationX_scaled, trainY, validationY, testX_scaled, testY

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: ./randomized.py <random_hill_climb OR simulated_annealing OR genetic_alg OR mimic>'''")
		exit(0)
	trainX, validationX, trainY, validationY, testX, testY = prepareData(0.2)

	algorithm = sys.argv[1]
	np.random.seed(3)
	models = []
	accs = []

	for i in range(1, 100, 10):
		nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = algorithm, max_iters = i,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100)

		nn_model1.fit(trainX, trainY)
	
		y_train_pred = nn_model1.predict(trainX)
		y_train_accuracy = accuracy_score(trainY, y_train_pred)
		y_validation_pred = nn_model1.predict(validationX)
		y_validation_accuracy = accuracy_score(validationY, y_validation_pred)

		models.append(nn_model1)
		accs.append(y_validation_accuracy)

		print('Iter: ', i, 'Training accuracy: ', y_train_accuracy, 'Validation accuracy: ', y_validation_accuracy)

	maxScore = max(accs)
	index = accs.index(maxScore)
	model = models[index]
	y_test_pred = model.predict(testX)
	y_test_accuracy = accuracy_score(testY, y_test_pred)

	print('Test accuracy: ', y_test_accuracy)
