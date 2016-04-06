#!/usr/bin/env python
import sys

import numpy as np
import matplotlib.pyplot as plt

from reference_algorithms import *
from world_objects import *
import pickle

algs_to_test = {'naive' : naive_algorithm, 
				'objects' : objects_algorithm, 
				'objects_walls' : objects_walls_algorithm,
				'refpt' : ow_refpt_algorithm}

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
def test_algorithms(data, commands, world, algorithms, plot=True, filename=None):
	# Cheating algorithm is special case
	probs = {"cheating" : get_cheating_prob(data)}
	for alg_name in algorithms:
		alg_probs = []
		for point_set in range(1, 13):
			# Calculate pdf
			dist = algorithms[alg_name](commands[point_set], world)

			# Sum of log prob = log of product prob
			log_prob = np.log(dist.pdf(data[point_set]))
			alg_probs.append(np.sum(log_prob))
		probs[alg_name] = np.array(alg_probs)

	if plot:
		fig, ax = plt.subplots()
		ind = np.arange(1, 13)
		width = 1/float(len(probs) + 1)
		colors = ['b', 'r', 'g', 'k', 'm']
		algs = ['cheating', 'naive', 'objects', 'objects_walls', 'refpt']
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

def gridsearch(data, commands, world):
	k1 = np.arange(4.1, 5, .1)
	k2 = np.arange(3.5, 4.2, .1)
	print k2
	L2 = np.zeros((len(k1), len(k2)))
	cheating_prob = get_cheating_prob(data)
	for i, val1 in enumerate(k1):
		for j, val2 in enumerate(k2):
			print float(len(k2)*i + j)/L2.size*100, "%"
			probs = np.array([np.sum(np.log(ow_refpt_algorithm(commands[num], world, val1, val2).pdf(data[num]))) for num in range(1, 13)])
			L2[i, j] = np.linalg.norm(cheating_prob - probs)
	print "100.0 %"
	loc = np.where(L2 == L2.min())
	print
	print 'k1 =', k1[loc[0][0]]
	print 'k2 =', k2[loc[1][0]]
	plt.contourf(k2, k1, L2)
	plt.show()
	return L2

def get_all_distributions(data, commands, world):
	global algs_to_test
	distributions = {}
	distributions['cheating'] = {i : cheating_algorithm(data[i]) for i in range(1, 13)}
	for alg in algs_to_test:
		distributions[alg] = {i : algs_to_test[alg](commands[i], world) for i in range(1, 13)}
	return distributions

def get_means(distributions):
	return {name : np.array([distributions[name][i].mean for i in range(1, 13)]) for name in distributions}

def eval_means(means, data=None, commands=None, filename=None):
	L2 = {}
	for name in means:
		if name == 'cheating':
			continue
		L2[name] = np.sqrt(np.sum((means['cheating'] - means[name])**2, axis=1))

	fig, ax = plt.subplots()
	ind = np.arange(1, 13)
	width = 1/float(len(L2) + 1)
	colors = ['r', 'g', 'k', 'm']
	algs = ['naive', 'objects', 'objects_walls', 'refpt']
	if data and commands:
		covs = {pt : np.cov(data[pt].T) for pt in data}
		var_parallel = [covs[i][0, 0] if commands[i].direction[0] else covs[i][1, 1] for i in range(1, 13)]
		stds = [np.sqrt(var) for var in var_parallel]
		ax.scatter(ind + 2*width, stds, c='b', marker='*', label='Empirical STD')
	for i, alg in enumerate(algs):
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

# I wonder if discrepency in command 7 is because of duct tape?
# Looks like there was significant overestimation in that case
# That's very possible, a lot of people in the second scene 1 trial expressed confusion about that - Eddie
if __name__ == '__main__':
	if len(sys.argv) > 1:
		scenenum = sys.argv[1]
		datafile = 'data/scene_' + scenenum + '_images_annotated_preprocessed.dat'
		scenenum = int(scenenum) - 1
		commands = all_commands[scenenum]
		world = all_worlds[scenenum]
	else:
		print "need scene number"
		sys.exit(1)
	with open(datafile) as dat:
		data = pickle.load(dat)
	if len(sys.argv) > 2 and sys.argv[2] == 'means':
		distributions = get_all_distributions(data, commands, world)
		means = get_means(distributions)
		if len(sys.argv) > 3 and sys.argv[3] == 'save':
			print "Saved"
			L2 = eval_means(means, data, commands, filename=sys.argv[4])
		else:
			L2 = eval_means(means, data, commands)
	else:
		if len(sys.argv) > 2 and sys.argv[2] == 'save':
			print "Saved"
			test_algorithms(data, commands, world, algs_to_test, filename=sys.argv[3])
		else:
			test_algorithms(data, commands, world, algs_to_test)

