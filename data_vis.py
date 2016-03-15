#!/usr/bin/env python

import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal

import sys

"""
Get the data by doing:

	from data_vis import load_data
	data = load_data('point_data.csv')

then data is a dictionary with keys 1-12, referring to the points
"""
def load_data(filename):
	text = None
	with open(filename) as f:
		text = f.readline()

	text = text.split('\r')
	data = {}
	for line in text:
		line = line.split(',', 1)
		point_num = int(line[0])
		point_loc = [float(el) for el in line[1][2:-2].split(',')] # Parsing the tuple is annoying
		data[point_num] = data.get(point_num, []) + [point_loc]

	for i in range(1, 13):
		data[i] = np.array(data[i])

	return data

# Get cheating normal distributions
def get_statistics(data):
	means = {pt : np.mean(data[pt], axis=0) for pt in data}
	covariances = {pt : np.cov(data[pt].T) for pt in data}
	cheating_normals = {pt : multivariate_normal(means[pt], covariances[pt]) for pt in data}
	return cheating_normals

def visualize(data, filename=None):
	fig, ax = plt.subplots()
	ax.set_xlim([0, 48]) # Set x dim to 4 feet
	ax.set_ylim([0, 36]) # Set y dim to 3 feet

	# Object dimensions from https://docs.google.com/document/d/1TbAKCrdEfgD6nCEjhlpJ4BpAJcmeSajsGQwb4HBsh7U/edit
	keyboard = Polygon([[33.375, 13.5], [38.5, 13.5], [38, 24.625], [33, 24.375]], True)
	car = Polygon([[9.5, 21.5], [10.625, 21.25], [11.625, 24], [10.375, 24.5]], True)
	bowl = Circle((19.75, 10), 2.5)

	objects = PatchCollection([keyboard, car, bowl])
	ax.add_collection(objects)

	colors = cm.rainbow(np.linspace(0, 1, 12))
	for i in range(1, 13):
		plt.scatter(data[i][:,0], data[i][:,1], c=colors[i-1])

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()


if __name__ == '__main__':
	data = load_data('point_data.csv')
	if len(sys.argv) == 3 and sys.argv[1] == 'save':
		visualize(data, sys.argv[2])
	else:
		visualize(data)