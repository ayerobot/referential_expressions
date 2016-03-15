#!/usr/bin/env python

import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal

import sys

# Distances Specified by each command
command_distances = {
	1 : np.array([-4, 0]),   # 4   inches left     of the car
	2 : np.array([-12, 0]),  # 1   foot   left     of the keyboard
	3 : np.array([18, 0]),   # 1.5 feet   right    of the car
	4 : np.array([0, -4]),   # 4   inches in front of the pink bowl
	5 : np.array([0, 5.5]),  # 5.5 inches behind      the keyboard
	6 : np.array([-2, 0]),   # 2   inches left     of the pink bowl
	7 : np.array([0, 1.5]),  # 1.5 inches behind      the car
	8 : np.array([7, 0]),    # 7   inches right    of the pink bowl
	9 : np.array([0, -16]),  # 16  inches in front of the car
	10 : np.array([0, 24]),  # 2   feet   behind      the pink bowl
	11 : np.array([3, 0]),   # 3   inches right    of the keyboard
	12 : np.array([0, -4.5]) # 4.5 inches in front of the keyboard
}

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

def get_means(data):
	return {pt : np.mean(data[pt], axis=0) for pt in data}

def get_covariances(data):
	return {pt : np.cov(data[pt].T) for pt in data}

"""
Get cheating normal distributions

Can use cheating_normals to get probability of data - 
	e.g. probs1 = cheating_normals[1].pdf(data[1])
"""
def get_cheating_distribution(data):
	means = get_means(data)
	covariances = get_covairances(data)
	cheating_normals = {pt : multivariate_normal(means[pt], covariances[pt]) for pt in data}
	return cheating_normals

def get_random_distribution(data):
	means = {pt : np.random.rand(2)*np.array(48, 36) for pt in data}
	random_normals = {pt : multivariate_normal(means[pt], np.array([1, 0], [0, 1])) for pt in data}
	return random_normals

def plot_distance_against_var_in_direction(data):
	fig, ax = plt.subplots()
	covariances = get_covariances(data)
	variances_in_direction = [covariances[i][0, 0] if command_distances[i][0] else covariances[i][1, 1] for i in range(1, 13)]
	distance = [np.linalg.norm(command_distances[i]) for i in range(1, 13)]
	plt.scatter(distance, variances_in_direction)
	ax.set_xlabel('Distance of Command (inches)')
	ax.set_ylabel('Variance in direction of command')
	plt.show()

def plot_distance_against_var_orthogonal_direction(data):
	fig, ax = plt.subplots()
	covariances = get_covariances(data)
	variances_in_direction = [covariances[i][1, 1] if command_distances[i][0] else covariances[i][0, 0] for i in range(1, 13)]
	distance = [np.linalg.norm(command_distances[i]) for i in range(1, 13)]
	plt.scatter(distance, variances_in_direction)
	ax.set_xlabel('Distance of Command (inches)')
	ax.set_ylabel('Variance orthogonal to direction of command')
	plt.show()

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

def visualize_distribution(points, distribution, filename=None):
	fig, ax = plt.subplots()
	ax.set_xlim([0, 48]) # Set x dim to 4 feet
	ax.set_ylim([0, 36]) # Set y dim to 3 feet

	x, y = np.mgrid[0:48:.1, 0:36:.1]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x
	pos[:, :, 1] = y
	plt.contourf(x, y, distribution.pdf(pos))

	# Object dimensions from https://docs.google.com/document/d/1TbAKCrdEfgD6nCEjhlpJ4BpAJcmeSajsGQwb4HBsh7U/edit
	keyboard = Polygon([[33.375, 13.5], [38.5, 13.5], [38, 24.625], [33, 24.375]], True)
	car = Polygon([[9.5, 21.5], [10.625, 21.25], [11.625, 24], [10.375, 24.5]], True)
	bowl = Circle((19.75, 10), 2.5)

	objects = PatchCollection([keyboard, car, bowl], cmap=matplotlib.cm.gray, alpha=1)
	objects.set_array(np.array([1, 1, 1]))
	ax.add_collection(objects)

	plt.scatter(points[:, 0], points[:, 1], c=np.array([1, 1, 1, 1]))

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()


if __name__ == '__main__':
	data = load_data('point_data.csv')
	if len(sys.argv) == 3 and sys.argv[1] == 'save':
		visualize(data, sys.argv[2])
	else:
		visualize(data)