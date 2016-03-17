import numpy as np
import matplotlib
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal

import sys

#scale factors derived from experimental data, determines relationship
#between distance of command and variance in the direction of the command
variance_scale = 0.43
variance_offset = -0.6

class Command:

	def __init__(self, sentence, distance, direction, reference):
		self.sentence = sentence
		self.distance = distance
		self.direction = direction #tuple specifying x and y direction (i.e, (0, 1), (0, -1), (1, 0), (-1, 0))
		self.reference = reference 

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
	def __init__(self, references, xdim, ydim):
		self.references=references
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


#takes in a command object and a world object, and returns a numpy.multivariate_norm distribution
#with random mean and fixed covariance 
def random_algorithm(cmd, world):
	randx = world.xdim*np.random.random_sample()
	randy = world.ydim*np.random.random_sample()

	mv_dist = multivariate_normal(np.array([randx, randy]), np.ndarray([[1, 0], [0, 1]]))

	return mv_dist


#takes in a command and a world object, and returns a multivariate normal distribution
#with mean equal to the "precise" point specified in the command
#with variance in the direction of the command equal to 0.43*distance - 0.6 (derived from experimental plot of distance vs variance)
def naive_algorithm(cmd, world):

	mean = estimate_pos(cmd)
	#variances in the command direction and in the other direction
	variance_cmd_direction = cmd.distance*variance_scale + variance_offset
	variance_cmd_ortho = 2 # inches hard-coding this for now, but there's definitely a relationship between it and something else

	if cmd.direction[0]: #if the command is in the x direction
		covar = np.ndarray([[variance_cmd_direction, 0], [0, variance_cmd_ortho]])
	else:
		covar = np.ndarray([[variance_cmd_ortho, 0], [0, variance_cmd_direction]])

	mv_dist = multivariate_normal(mean, covar)




