#!/usr/bin/env python2

import sys

import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal

from world import *
from reference_algorithms import cheating_algorithm, random_algorithm, naive_algorithm, naive_algorithm2, objects_walls_algorithm





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

def plot_distance_parallel(data, filename=None):
	fig, ax = plt.subplots()
	covariances = get_covariances(data)
	variances_in_direction = [covariances[i][0, 0] if commands[i].direction[0] else covariances[i][1, 1] for i in range(1, 13)]
	distance = [commands[i].distance for i in range(1, 13)]
	#calculating line of best fit
	line_bestfit = np.polyfit(distance, variances_in_direction, 1)
	print "Linear fit results: slope= " + str(line_bestfit[0]) + " intercept = " + str(line_bestfit[1]) 
	#plotting line of best fit
	distance_range = np.linspace(0, 30)
	bestfit_points = distance_range*line_bestfit[0] + line_bestfit[1]
	plt.plot(distance_range, bestfit_points, label="var = 0.43*dist - 0.59")
	plt.legend()
	plt.scatter(distance, variances_in_direction)
	ax.set_xlabel('Distance of Command (inches)')
	ax.set_ylabel('Variance in direction of command')

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

def plot_distance_orthogonal(data):
	fig, ax = plt.subplots()
	covariances = get_covariances(data)
	variances_in_direction = [covariances[i][1, 1] if commands[i].direction[0] else covariances[i][0, 0] for i in range(1, 13)]
	distance = [commands[i].distance for i in range(1, 13)]
	plt.scatter(distance, variances_in_direction)
	ax.set_xlabel('Distance of Command (inches)')
	ax.set_ylabel('Variance orthogonal to direction of command')
	plt.show()

# i = command number
def estimated_position(i):
	ref = commands[i].reference
	center = ref.center
	direction = commands[i].direction
	vector = commands[i].distance*direction
	offset = ref.width/2.*direction if direction[0] else ref.height/2.*direction
	return center + offset + vector

def visualize(data, world, filename=None):
	fig, ax = plt.subplots()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	objects = PatchCollection([ref.patch for ref in world.references])
	objects.set_array(np.array([0, 0, 0, 1]))
	ax.add_collection(objects)

	colors = cm.rainbow(np.linspace(0, 1, 12))
	for i in range(1, 13):
		plt.scatter(data[i][:,0], data[i][:,1], c=colors[i-1], marker='+')
		estimated_pos = estimated_position(i)
		plt.scatter(estimated_pos[0], estimated_pos[1], c=np.array([0, 0, 0, 1]), marker='o')

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

def visualize_distribution(points, distribution, world, filename=None, block=True):
	ax = plt.gca()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x
	pos[:, :, 1] = y
	plt.contourf(x, y, distribution.pdf(pos))

	objects = PatchCollection([ref.patch for ref in world.references], cmap=cm.gray, alpha=1)
	objects.set_array(np.array([1, 1, 1]))
	ax.add_collection(objects)

	plt.scatter(points[:, 0], points[:, 1], c=np.array([1, 1, 1, 1]), marker='.')

	if filename:
		plt.savefig(filename, format='pdf')

	if block:
		plt.show()

# algorithm is a string, either 'cheating', 'random', or 'naive'
def visualize_all_distributions(data, commands, algorithm, world, filename=None):
	fig = plt.figure(figsize=(18, 8))
	fig.subplots_adjust(hspace=0.5)
	distributions = None
	algorithms = {"random" : random_algorithm, "naive" : naive_algorithm, "naive2" : naive_algorithm2, "walls" : objects_walls_algorithm}
	if algorithm == "cheating":
		distributions = {pts : cheating_algorithm(data[pts]) for pts in data}
	elif algorithm in algorithms:
		distributions = {cmdnum : algorithms[algorithm](commands[cmdnum], world) for cmdnum in commands}
	else:
		raise ValueError("Unknwon Algorithm: " + str(algorithm))

	for i in range(1, 13):
		plt.subplot(3, 4, i)
		visualize_distribution(data[i], distributions[i], world, block=False)
		plt.title(commands[i].sentence, fontsize=12)

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

if __name__ == '__main__':
	data = load_data('point_data.csv')
	if len(sys.argv) == 3 and sys.argv[1] == 'save':
		visualize(data, sys.argv[2])
	else:
		visualize(data, world)