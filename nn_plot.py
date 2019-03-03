#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

if __name__ == '__main__':
	algs = ["random_hill_climbing", "simulated_annealing", "genetic"]
	output = "nn_plot.jpg"

	x = []
	yTrain = []
	yTest = []
	for algo in algs:
		filename = "./out/" + algo + ".out"
		curX = []
		curYTrain = []
		curYTest = []
		with open(filename,"r") as f:
			reader = csv.reader(f, delimiter=",", skipinitialspace=True)
			for i, line in enumerate(reader):
				curX.append(int(line[0]))
				curYTrain.append(float(line[1]))
				curYTest.append(float(line[2]))
		x.append(curX)
		yTrain.append(curYTrain)
		yTest.append(curYTest)

	plt.figure(figsize=(16,16))
	axes = plt.gca()
	train_line = ["b:","r:","g:"]
	test_line = ["b-","r-","g-"]
	for algo in algs:
		index = algs.index(algo)
		line1, = plt.plot(x[index],yTrain[index],train_line[index],linewidth=1,label=algo + "_train")
		line2, = plt.plot(x[index],yTest[index],test_line[index],linewidth=1,label=algo + "_test")


	# val, = plt.plot(x[2],y[2],plot_line[2],linewidth=1,label="RHC")


	plt.xlabel("Iteration")
	plt.ylabel("Accuracy") 
	plt.legend()
	plt.savefig(output)