#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: ./ploy.py <problem_name>")
	problem_name = sys.argv[1]
	algs = ["GA", "SA", "RHC", "Mimic"]
	output = problem_name + "_plot.jpg"

	x = []
	y = []
	for algo in algs:
		filename = problem_name + algo + ".out"
		curX = []
		curY = []
		with open(filename,"r") as f:
			reader = csv.reader(f, delimiter=",", skipinitialspace=True)
			for i, line in enumerate(reader):
				curX.append(int(line[0]))
				curY.append(float(line[1]))
		x.append(curX)
		y.append(curY)

	plt.figure(figsize=(16,16))
	axes = plt.gca()
	plot_line = ["b-","r-","g-","p-"]
	for algo in algs:
		index = algs.index(algo)
		val, = plt.plot(x[index],y[index],plot_line[index],linewidth=1,label=algo)

	# val, = plt.plot(x[2],y[2],plot_line[2],linewidth=1,label="RHC")


	plt.xlabel("Iteration")
	plt.ylabel("Fitness") 
	plt.legend()
	plt.savefig(output)