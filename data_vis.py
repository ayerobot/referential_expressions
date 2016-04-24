#!/usr/bin/env python
import sys

import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
import pickle

from world_objects import *
from reference_algorithms import *
from algorithm_evaluation import algs_to_test
from data_utils import *


"""
Get the data by doing:

	from data_vis import load_data
	data = load_data('point_data.csv')

then data is a dictionary with keys 1-12, referring to the points
"""

def get_means(data):
	return {pt : np.mean(data[pt], axis=0) for pt in data}

def get_covariances(data):
	return {pt : np.cov(data[pt].T) for pt in data}

def plot_distance_parallel(data, commands, filename=None):
	fig, ax = plt.subplots()
	covariances = get_covariances(data)
	variances_in_direction = [covariances[i][0, 0] if commands[i].direction[0] else covariances[i][1, 1] for i in range(1, len(data) + 1)]
	distance = np.array([commands[i].distance for i in range(1, len(data) + 1)])
	#calculating line of best fit
	degree = 1
	line_bestfit = np.polyfit(distance, variances_in_direction, degree)
	equation = str(line_bestfit[0]) + "x + " + str(line_bestfit[1])
	print "Linear fit results: " + equation
	#plotting line of best fit
	distance_range = np.linspace(0, distance.max())
	bestfit_points = np.ones(distance_range.shape)
	for i in range(degree + 1):
		bestfit_points  = bestfit_points + (distance_range**(degree - i))*line_bestfit[i]
	bestfit_points2 = distance_range*0.43 - 0.59
	plt.plot(distance_range, bestfit_points, label=equation)
	plt.plot(distance_range, bestfit_points2, label='old equation')
	plt.legend()
	plt.scatter(distance, variances_in_direction)
	ax.set_xlabel('Distance of Command (inches)')
	ax.set_ylabel('Variance in direction of command')

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

def plot_distance_orthogonal(data, commands):
	fig, ax = plt.subplots()
	covariances = get_covariances(data)
	variances_in_direction = [covariances[i][1, 1] if commands[i].direction[0] else covariances[i][0, 0] for i in range(1, len(data))]
	distance = [commands[i].distance for i in range(1, len(data))]
	plt.scatter(distance, variances_in_direction)
	ax.set_xlabel('Distance of Command (inches)')
	ax.set_ylabel('Variance orthogonal to direction of command')
	plt.show()

def visualize(data, world, commands, filename=None):
	fig, ax = plt.subplots()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	objects = PatchCollection([ref.patch for ref in world.references])
	objects.set_array(np.array([0, 0, 0, 1]))
	ax.add_collection(objects)


	colors = cm.rainbow(np.linspace(0, 1, 12))
	markers = ['+', 'o', 'v', '>', 's', 'p']
	for i in range(1, 13):
		plt.scatter(data[i][:,0], data[i][:,1], c=colors[i-1], marker=markers[(i - 1) % 6], label=str(i))
		estimated_pos = estimate_pos(commands[i])
		plt.scatter(estimated_pos[0], estimated_pos[1], c=np.array([0, 0, 0, 1]), marker='o')
		plt.text(estimated_pos[0] + 0.25, estimated_pos[1] + 0.25, str(i))

	plt.legend()

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

def visualize_distribution(points, distribution, world, filename=None, block=True):
	ax = plt.gca()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]
	plt.contourf(x, y, distribution.pdf(np.dstack((x, y))))

	objects = PatchCollection([ref.patch for ref in world.references], cmap=cm.gray, alpha=1)
	objects.set_array(np.array([1, 1, 1]))
	ax.add_collection(objects)

	plt.scatter(points[:, 0], points[:, 1], c=np.array([1, 1, 1, 1]), marker='.')

	if filename:
		plt.savefig(filename, format='pdf')

	if block:
		plt.show()

"""
	elif algorithm == "loglin":
		weights = np.array([[0.7302, 0.1143, -0.6606],
    						[0.7637, 0.2605, 0.1052],
						    [0.3115, 0.0263, 0.2983],
						    [1.6282, 0.0824, -0.5241],
						    [1.8234, 0.0813, 1.2681],
						    [1.0695, 0.0824, 0.1040],
						    [0.4629, 0.1143, 0.1314],
						    [0.8889, 0.4657, -3.4583],
						    [0.4676, 0.5182, 0.0042],
						    [0.3888, 0.3969, 0.1461],
						    [0.6784, 0.0813, 0.0538],
						    [1.2481, 0.0813, -2.9671]])
		distributions = {cmdnum : loglin_alg(commands[cmdnum], world, weights[cmdnum-1,:]) for cmdnum in commands}
	"""
# algorithm is a string, either 'cheating', 'random', or 'naive'
def visualize_all_distributions(data, commands, algorithm, world, filename=None):
	fig = plt.figure(figsize=(18, 8))
	fig.subplots_adjust(hspace=0.5)
	distributions = None
	if algorithm == "cheating":
		distributions = {pts : cheating_algorithm(data[pts]) for pts in data}

	elif algorithm in algs_to_test:
		distributions = {cmdnum : algs_to_test[algorithm](commands[cmdnum], world) for cmdnum in commands}
	else:
		raise ValueError("Unknown Algorithm: " + str(algorithm))

	for i in range(1, 13):
		plt.subplot(3, 4, i)
		visualize_distribution(data[i], distributions[i], world, block=False)
		plt.title(commands[i].sentence, fontsize=12)

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

if __name__ == '__main__':
	if len(sys.argv) > 1:
		scenenum = sys.argv[1]
		datafile = 'data/scene_' + scenenum + '_images_annotated_preprocessed.dat'
		scenenum = int(scenenum)
		commands = all_commands[scenenum]
		world = all_worlds[scenenum]
	else:
		print "need scene number"
		sys.exit(1)
	with open(datafile) as dat:
		data = pickle.load(dat)
	if len(sys.argv) > 2 and sys.argv[2] != 'save':
		if len(sys.argv) > 3 and sys.argv[3] == 'save':
			print "Saved"
			visualize_all_distributions(data, commands, sys.argv[2], world, sys.argv[4])
		else:
			visualize_all_distributions(data, commands, sys.argv[2], world)
	elif len(sys.argv) == 4 and sys.argv[2] == 'save':
		print "Saved"
		visualize(data, world, commands, sys.argv[3])
	else:
		visualize(data, world, commands)