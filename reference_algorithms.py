import numpy as np
import matplotlib
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal, expon
from parser import parse

import sys

#scale factors derived from experimental data, determines relationship
#between distance of command and variance in the direction of the command
variance_scale = 0.43
variance_offset = -0.6

#map natural language direction to direction tuples:
lang_to_tup = {"right of":np.array([1, 0]), "left of":np.array([-1, 0]), "to the left of":np.array([-1, 0]), "to the right of":np.array([1, 0]), "in front of":np.array([0, -1]), "behind":np.array([0, 1])}

class Command:

	def __init__(self, sentence, world):
		self.sentence = sentence
		#parsing sentences
		num, unit, direction, reference = parse(sentence)
		self.direction = lang_to_tup[direction]
		self.distance = num #autoconverted to inches
		#finding the reference in the world
		matching_references = [ref for ref in world.references if ref.name == reference]
		self.reference = matching_references[0] 

	def __repr__(self):
		return self.sentence

class Reference:

	def __init__(self, name, position):
		self.name = name
		if isinstance(position, tuple):
			self.position = position[0]
			self.center = position[0]
			self.radius = position[1]
			self.height = 2*self.radius
			self.width = 2*self.radius
			self.patch = Circle(self.center, self.radius)
		else:
			self.position = position
			self.center = np.mean(self.position, axis=0)
			self.width = self.position[:,0].ptp()
			self.height = self.position[:,1].ptp()
			self.patch = Polygon(self.position, True)

class World:
	#dimensions in inches
	#coordinates will be relative to bottom left corner
	#references will be a list of world objects (TODO: possibly a set or a dict?)
	def __init__(self, references, xdim, ydim):
		self.references= references
		self.xdim = xdim
		self.ydim = ydim

	#future: instance methods for editing reference list, including type-checking
	#def add_reference(self, ref)

#TODO: create generalized "result object"
#all results from algorithm functions will return a function that has a .pdf method

#estimates the "exact" position that a command is referring to 
def estimate_pos(cmd):
	ref = cmd.reference
	center = ref.center
	direction = cmd.direction
	vector = cmd.distance*direction
	offset = ref.width/2.*direction if direction[0] else ref.height/2.*direction
	return center + offset + vector

# takes in training data set for a given command and returns a numpy.multivariate_norm distribution
# with the sample mean and covariance
def cheating_algorithm(pts):
	mean = np.mean(pts, axis=0)
	covariance = np.cov(pts.T)

	mv_dist = multivariate_normal(mean, covariance)

	return mv_dist

#takes in a command object and a world object, and returns a numpy.multivariate_norm distribution
#with random mean and fixed covariance 
def random_algorithm(cmd, world):
	randx = world.xdim*np.random.random_sample()
	randy = world.ydim*np.random.random_sample()

	mv_dist = multivariate_normal(np.array([randx, randy]), np.array([[1, 0], [0, 1]]))

	return mv_dist


#takes in a command and a world object, and returns a multivariate normal distribution
#with mean equal to the "precise" point specified in the command
#with variance in the direction of the command equal to 0.43*distance - 0.6 (derived from experimental plot of distance vs variance)
def naive_algorithm(cmd, world):
	mean = estimate_pos(cmd)
	#variances in the command direction and in the other direction

	variance_cmd_parallel = max(cmd.distance*variance_scale + variance_offset, 0.5)
	variance_cmd_ortho = 0.5 # inches hard-coding this for now, but there's definitely a relationship between it and something else

	if cmd.direction[0]: #if the command is in the x direction
		covar = np.array([[variance_cmd_parallel, 0], [0, variance_cmd_ortho]])
	else:
		covar = np.array([[variance_cmd_ortho, 0], [0, variance_cmd_parallel]])

	mv_dist = multivariate_normal(mean, covar)

	return mv_dist

# Estimate new mean position
# Find argmax_{x, y} naive_algorithm(cmd, world)*1/scale*e^{||(x, y) - (x_ref, y_ref)|| - min_{obj}||(x, y) - (x_obj, y_obj)||/scale}
def naive_algorithm2(cmd, world):
	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]

	# Calculate mean
	naive_dist = naive_algorithm(cmd, world)
	mean_vals = naive_dist.pdf(np.dstack((x, y)))

	# Find Distance to closest object
	ref_dists = {ref : np.sqrt((x - ref.center[0])**2 + (y - ref.center[1])**2) for ref in world.references}
	min_dists = np.min(np.dstack(ref_dists[ref] for ref in ref_dists), axis=2)

	# Difference between distance to closest object and object reference in command
	distance_diff = ref_dists[cmd.reference] - min_dists

	exp_vals = expon.pdf(distance_diff, scale=0.7)

	vals = mean_vals*exp_vals
	vals = vals
	loc = np.where(vals == vals.max())
	mean = 0.1*np.array([loc[0][0], loc[1][0]])

	mv_dist = multivariate_normal(mean, naive_dist.cov)

	return mv_dist

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
	e.g. {"naive" : naive_algorithm, "naive2" : naive_algorithm2}

Assumes that all algorithms (except cheating) take in only two arguments, command and world
"""
def test_algorithms(data, commands, world, algorithms, plot=True):
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
		for alg in probs:
			plt.plot(np.arange(1, 13), probs[alg], label=alg)
		ax = plt.gca()
		ax.set_xlim([1, len(commands)])
		ax.set_xlabel('Command Number')
		ax.set_ylabel('log prob of product of data')
		plt.legend()
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













