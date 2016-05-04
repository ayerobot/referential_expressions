#!/usr/bin/env python
import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import data_utils
from utils.world_objects import *
from reference_algorithms import *
from loglin_algorithm import LoglinDistribution, loglin_distributions

algs_to_test = {'naive' : naive_algorithm, 
				'objects' : objects_algorithm, 
				'objects_walls' : objects_walls_algorithm,
				'refpt' : ow_refpt_algorithm,
				'loglin' : loglin_alg}

"""
Cheating algorithm is a bit of a special case since it doesn't take in command, world. 
Therefore this is a special method just to calculate it.
"""
def get_cheating_prob(data):
	cheat_prob = []
	for point_set in range(1, 13):
		cheat_dist = cheating_algorithm(data[point_set])

		cheat_prob.append(np.sum(np.log(cheat_dist.pdf(data[point_set]))))

	cheat_prob = np.array(cheat_prob)
	return cheat_prob

"""
For each command, get log-probability of product of all data points

algorithms is a dictionary of the form {<algorithm name> : <function>}
	e.g. {"naive" : naive_algorithm, "objects" : objects_algorithm}

Assumes that all algorithms (except cheating) take in only two arguments, command and world
"""

#I think distributions already has this information?
def test_algorithms(data, commands, world, distributions, algs, plot=True, filename=None):
	# Cheating algorithm is special case
	probs = {"cheating" : get_cheating_prob(data)}
	for alg_name in distributions:
		alg_probs = []
		for point_set in range(1, 13):
			# Calculate pdf
			#dist = algorithms[alg_name](commands[point_set], world)
			dist = distributions[alg_name][point_set]

			# Sum of log prob = log of product prob
			log_prob = np.log(dist.pdf(data[point_set]))
			alg_probs.append(np.sum(log_prob))
		probs[alg_name] = np.array(alg_probs)

	if plot:
		fig, ax = plt.subplots()
		ind = np.arange(1, 13)
		width = 1/float(len(probs) + 1)
		colors = ['b', 'y', 'g', 'k', 'm', 'r']
		#algs = ['cheating', 'naive', 'objects', 'objects_walls', 'refpt', 'loglin']
		for i, alg in enumerate(algs):
			ax.bar(ind + i*width, -probs[alg], width, color=colors[i], label=alg)
		# add some text for labels, title and axes ticks
		ax.set_xlabel('Command Number')
		ax.set_ylabel('-log prob of product of data')
		ax.set_xlim([1, 13])
		ax.set_xticks(ind + 2.5*width)
		ax.set_xticklabels(ind)

		plt.legend(loc='upper center')
		if filename:
			plt.savefig(filename, format='pdf')
		plt.show()
	else:
		return probs

def calculate_logprob(data, commands, world, algs, filename=None):
	probs = test_algorithms(data, commands, world, algs_to_test, False, filename)

	probs = {name : np.sum(probs[name]) for name in probs}

	fig, ax = plt.subplots()
	width = 1/float(len(probs))
	#algs = ['cheating', 'naive', 'objects', 'objects_walls', 'refpt', 'loglin']
	for i, alg in enumerate(algs):
		ax.bar(i*width, -probs[alg], width, color='b', label=alg)

	ax.set_xlabel('Algorithms')
	ax.set_ylabel('-log prob of product of data')
	mid_points = width/2 + np.arange(len(probs))*width
	ax.set_xticks(mid_points)
	ax.set_xticklabels(algs)

	if filename:
		plt.savefig(filename, format='pdf')
	plt.show()

"""
Takes in result of test_algorithms. Calculates difference between each distribution and
the cheating distribution. 
"""
def eval_algorithms(algorithm_probs):
	diffs = {}
	for alg_name in algorithm_probs:
		if alg_name == 'cheating':
			continue
		diffs[alg_name] = algorithm_probs['cheating'] - algorithm_probs[alg_name]
	L1 = {alg_name : np.sum(diffs[alg_name]) for alg_name in diffs}
	L2 = {alg_name : np.linalg.norm(diffs[alg_name]) for alg_name in diffs}
	print "L1 Norms: ", L1
	print "L2 Norms: ", L2
	return diffs

def get_all_distributions(data, commands, world, algs_to_test):
	#global algs_to_test
	distributions = {}
	distributions['cheating'] = {i : cheating_algorithm(data[i]) for i in range(1, 13)}
	for alg in algs_to_test:
		distributions[alg] = {i : algs_to_test[alg](commands[i], world) for i in range(1, 13)}
	return distributions

#gets all log-linear distributions for each set of features
def get_all_loglin(data, commands, world, loglin_data):
	distributions = {}
	distributions['cheating'] = {i : cheating_algorithm(data[i]) for i in range(1, 13)}
	for name in loglin_data:
		feats, weights = loglin_data[name]
		distributions[name] = {i: LoglinDistribution(commands[i], world, w=weights, feats=feats) for i in range(1, 13)}
	return distributions


def get_means(distributions):
	return {name : np.array([distributions[name][i].mean for i in range(1, 13)]) for name in distributions}

#returns a dictionary for each algorithm
#each element in the dictionary is a list
#each element in the list corresponds to a command, contains a numpy array of the errors (NOT squared errors) for every point in that command 
def get_error(data, means):
	error = {name : [] for name in means}
	for name in means:
		for i in range(12):
			sq_err = (data[i + 1] - means[name][i])**2
			error[name].append(np.sqrt(np.sum(sq_err, 1))) #array of distances from the predicted mean for each data point, i.e. errors
	return error

def eval_means(means, data=None, commands=None, filename=None):
	L2 = {}
	for name in means:
		if name == 'cheating':
			continue
		L2[name] = np.sqrt(np.sum((means['cheating'] - means[name])**2, axis=1))

	fig, ax = plt.subplots()
	ind = np.arange(1, 13)
	width = 1/float(len(L2) + 1)
	colors = ['y', 'g', 'k', 'm', 'r']
	if data and commands:
		covs = {pt : np.cov(data[pt].T) for pt in data}
		var_parallel = [covs[i][0, 0] if commands[i].direction[0] else covs[i][1, 1] for i in range(1, 13)]
		stds = [np.sqrt(var) for var in var_parallel]
		ax.scatter(ind + 2*width, stds, c='b', marker='*', label='Empirical STD')
	for i, alg in enumerate(algs_to_test):
		ax.bar(ind + i*width, L2[alg], width, color=colors[i], label=alg)
	ax.set_xlim([1, 13])
	ymin, ymax = ax.get_ylim()
	ax.set_ylim([0, ymax])
	ax.set_xticks(ind + 2*width)
	ax.set_xticklabels(ind)
	ax.set_xlabel('Command Number')
	ax.set_ylabel('Distance (in)')
	plt.legend(loc='upper center')
	plt.title('Distance between estimated and empirical means')
	if filename:
		plt.savefig(filename, format='pdf')
	plt.show()
	return L2

#plots the MSE values for a single command for all algorithms
def MSE_command(commands, cmd_num, errors, algs, block=True, filename=None):
	ax = plt.gca()

	MSE_command = {name: np.mean(errors[name][cmd_num - 1]**2) for name in errors} #gets the mean squared error for this command
	confidence_interval = {name: 2*np.std(errors[name][cmd_num - 1]**2) for name in errors} #twice the standard deviation of the squared errors

	colors = ['b', 'y', 'g', 'k', 'm', 'r']
	width = 1/float(len(MSE_command))
	#algs = ['cheating', 'naive', 'objects', 'objects_walls', 'refpt', 'loglin']
	for i, name in enumerate(algs):
		ax.bar(i*width, MSE_command[name], width, color='b', label=name, yerr=confidence_interval[name], ecolor='r')
	ymin, ymax = ax.get_ylim()
	mid_points = width/2 + np.arange(len(MSE_command))*width
	ax.set_ylim([0, ymax])
	ax.set_xticks(mid_points)
	ax.set_xticklabels(algs)
	ax.set_xlabel('Algorithm')
	ax.set_ylabel('MSE')
	plt.title(commands[cmd_num].sentence)

	if block:
		plt.show()


#plots the MSE by each command
def MSE_all_commands(data, commands, world, distributions, means, algs, filename=None):
	fig = plt.figure(figsize=(18, 8))
	fig.subplots_adjust(hspace=0.5)

	#calculate errors 
	errors = get_error(data, means)

	for i in range(1, 13):
		plt.subplot(3, 4, i)
		MSE_command(commands, i, errors, algs, block=False)

	plt.show()


def calculate_MSE(data, commands, world, distributions, means, algs, filename=None):
	error = get_error(data, means)
	MSE = {name : np.mean(np.hstack(error[name])**2) for name in error} #mean of the squared error
	confidence_interval = {name : 2*(np.hstack(error[name])**2).std() for name in error} #twice standard deviation of squared error
	fig, ax = plt.subplots()
	colors = ['b', 'y', 'g', 'k', 'm', 'r']
	width = 1/float(len(MSE))
	#algs = ['cheating', 'naive', 'objects', 'objects_walls', 'refpt', 'loglin']
	for i, name in enumerate(algs):
		ax.bar(i*width, MSE[name], width, color='b', label=name, yerr=confidence_interval[name], ecolor='r')
	ymin, ymax = ax.get_ylim()
	mid_points = width/2 + np.arange(len(MSE))*width
	ax.set_ylim([0, ymax])
	ax.set_xticks(mid_points)
	ax.set_xticklabels(algs)
	ax.set_xlabel('Algorithm')
	ax.set_ylabel('MSE')
	if filename:
		plt.savefig(filename, format='pdf')
	plt.show()
	return MSE

# I wonder if discrepency in command 7 is because of duct tape?
# Looks like there was significant overestimation in that case
# That's very possible, a lot of people in the second scene 1 trial expressed confusion about that - Eddie
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
	data = data_utils.load_pickle_data(datafile)
	#distributions = get_all_distributions(data, commands, world, algs_to_test)
	distributions = get_all_loglin(data, commands, world, loglin_distributions)
	means = get_means(distributions)
	algs = ['cheating', 'naive', 'naive_dist', 'naive_objects', 'naive_objects_walls']
	#algs = ['cheating', 'naive', 'objects', 'objects_walls', 'refpt', 'loglin']

	#TODO: fix interface

	if len(sys.argv) > 2 and sys.argv[2] == 'means':
		if len(sys.argv) > 3 and sys.argv[3] == 'save':
			print "Saved"
			L2 = eval_means(means, data, commands, algs, filename=sys.argv[4])
		else:
			L2 = eval_means(means, data, commands)
	elif len(sys.argv) > 2 and (sys.argv[2] == 'MSE' or sys.argv[2] == 'mse'):
		if len(sys.argv) > 3 and sys.argv[3] == 'save':
			print "Saved"
			MSE = calculate_MSE(data, commands, world, distributions, means, algs, filename=sys.argv[4])
		else:
			MSE = calculate_MSE(data, commands, world, distributions, means, algs)
			#MSE_all_commands(data, commands, world, distributions, means, algs)
	else:
		if len(sys.argv) > 2 and sys.argv[2] == 'save':
			print "Saved"
			test_algorithms(data, commands, world, distributions, algs, filename=sys.argv[3])
		else:
			test_algorithms(data, commands, world, distributions, algs)

