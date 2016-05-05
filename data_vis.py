#!/usr/bin/env python
import sys

import numpy as np
import numpy.matlib
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal

from utils import data_utils
from utils.world_objects import *
from reference_algorithms import *
from algorithm_evaluation import algs_to_test
from data_utils import *
from loglin_algorithm import LoglinDistribution, loglin_distributions

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
	#plt.plot(distance_range, bestfit_points2, label='old equation')
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

#plotting 4 commands on separate subplots
def visualize_commands(data, world, commands, command_numbers):
	plt.figure(figsize=(12, 9))
	for i in range(0, 4):
		plt.subplot(2, 2, i + 1)
		print "plotting command number " + str(command_numbers[i])
		visualize_command(data, world, commands, command_numbers[i], block=False)

	plt.show()

#only plots points from a single command, then labels it by the subject number
def visualize_command(data, world, commands, cmd_num, block = True, nums = False):
	ax = plt.gca()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	objects = PatchCollection([ref.patch for ref in world.references])
	objects.set_array(np.array([0, 0, 0, 1]))
	ax.add_collection(objects)
	cmd = commands[cmd_num]

	points_x = data[cmd_num][:,0]
	points_y = data[cmd_num][:,1]

	if nums:
		for i in range(0, len(points_x)):
			plt.text(points_x[i], points_y[i], str(i + 1))
	else:
		plt.scatter(points_x, points_y, color='red')

	plt.title(commands[cmd_num].sentence)

	if block:
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

# algorithm is a string, either 'cheating', 'random', or 'naive'
#loglin algorithms are stored in a different way, easier to just create a new function
def visualize_all_loglin(data, commands, feature_name, world, filename=None):
	fig = plt.figure(figsize=(18, 8))
	fig.subplots_adjust(hspace=0.5)
	distributions = None
	if feature_name in loglin_distributions:
		feats, weights = loglin_distributions[feature_name]
		distributions = {cmdnum : LoglinDistribution(commands[cmdnum], world, w=weights, feats=feats) for cmdnum in commands}
	else:
		raise ValueError("Unknown Feature Set: " + str(algorithm))

	for i in range(1, 13):
		plt.subplot(3, 4, i)
		visualize_distribution(data[i], distributions[i], world, block=False)
		plt.title(commands[i].sentence, fontsize=12)

	if filename:
		plt.savefig(filename, format='pdf')

	plt.show()

#plots all loglin distributions
def compare_all_loglin(data, commands, cmdnum, world, names):
	fig = plt.figure(figsize=(12, 9))
	fig.subplots_adjust(hspace=0.5)

	fig.suptitle(commands[cmdnum].sentence)

	for i, feature_name in enumerate(names):
		plt.subplot(2, 2, i + 1)
		feats, weights = loglin_distributions[feature_name]
		visualize_distribution(data[cmdnum], LoglinDistribution(commands[cmdnum], world, w=weights, feats=feats), world, block=False)
		plt.title(feature_name, fontsize=12)
	plt.show()


#visualizer for all multipeak loglin distributions
def visualize_all_multipeak(data, commands, world, filename=None):
	fig = plt.figure(figsize=(18, 8))
	fig.subplots_adjust(hspace=0.5)
	distributions = {}

	for cmdnum in range(1, 13):
		naive_means, directions, probs = get_naive_means(commands[cmdnum])
		distributions[cmdnum] = MultiPeakLoglin(commands[cmdnum], directions, naive_means, probs, world)

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

	#TODO: fix this interface + document it 
	with open(datafile) as dat:
		data = data_utils.load_pickle_data(datafile)
		if len(sys.argv) > 2 and sys.argv[2] != 'save':
			if len(sys.argv) > 3 and sys.argv[3] == 'save':
				print "Saved"
				visualize_all_distributions(data, commands, sys.argv[2], world, sys.argv[4])
			elif sys.argv[2] == 'loglin':
				if sys.argv[3] == 'compare':
					names = ['naive', 'naive_dist', 'naive_objects', 'naive_objects_walls'] #to make sure the names stay in the same order
					compare_all_loglin(data, commands, int(sys.argv[4]), world, names)
				else:
					visualize_all_loglin(data, commands, sys.argv[3], world)
			else:
				visualize_all_distributions(data, commands, sys.argv[2], world)
		elif len(sys.argv) == 4 and sys.argv[2] == 'save':
			print "Saved"
			visualize(data, world, commands, sys.argv[3])
		else:
			#visualize(data, world, commands)
			visualize_command(data, world, commands, 11)
