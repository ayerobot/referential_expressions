#!/usr/bin/env python2
import sys

import numpy as np
import scipy.io as sio

from utils.data_utils import *
from utils.world_objects import *
from reference_algorithms import *
from loglin_algorithm import *

"""
Performs a gridsearch over three weights for loglin algorithm.

Calculates joint probability of all data points given the algorithm.
"""
def gridsearch(data, commands, world):
	k1 = np.arange(0.5, 0.7, .01)
	k2 = np.arange(0.1, 0.25, .01)
	k3 = np.arange(0.2, 0.3, .01)
	all_probs = np.zeros((len(k1), len(k2), len(k3)))
	loglin_dists = [loglin_alg(commands[num], world) for num in range(1, 13)]
	for i, val1 in enumerate(k1):
		for j, val2 in enumerate(k2):
			for k, val3 in enumerate(k3):
				print float(len(k2)*len(k3)*i + len(k3)*j + k)/all_probs.size*100, "%"
				for dist in loglin_dists:
					dist.set_weights([val1, val2, val3])
				probs = [dist.pdf(data[num], normalize=True) for dist, num in zip(loglin_dists, range(1, 13))]
				logsum = np.sum(np.log(probs))
				all_probs[i, j, k] = logsum
	print "100.0 %"
	loc = np.where(all_probs == all_probs.max())
	print
	print 'k1 =', k1[loc[0][0]]
	print 'k2 =', k2[loc[1][0]]
	print 'k3 =', k3[loc[2][0]]
	return all_probs

"""
Saves feature vectors and average feature values for all commands for 
a given world.

Saves 12 files of the form 'grad_descent/command[1-12].mat'

Can load the files from the matlab code in grad_descent.

Each file contains:
	Tx => an (n x 3) matrix, where each row is the value of the features
			at a given data point. Every point in the discretization of the board is
			represented here.
	theta => a (1 x 3) vector containing the average value of the features at each
				data point.

This information is used to optimize the weights using the matlab code in the grad_descent
folder.
"""
def generate_feature_vectors(data, commands, world, features = feats_objects):
	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]
	for i in range(1, 13):
		cmd = commands[i]
		Tx = get_feature_matrix(x, y, cmd, cmd.direction, estimate_pos(cmd), world, features)
		theta = np.mean(get_feature_matrix(data[i][:,0], data[i][:,1], cmd, cmd.direction, estimate_pos(cmd), world, features), 0)
		sio.savemat('grad_descent/command' + str(i) + '.mat', {'Tx' : Tx, 'theta' : theta})

if __name__ == '__main__':
	scene_num = sys.argv[1]
	data = load_pickle_data('data/scene_' + scene_num + '_images_annotated_preprocessed.dat')
	scene_num = int(scene_num)
	commands = all_commands[scene_num]
	world = all_worlds[scene_num]
	generate_feature_vectors(data, commands, world)
