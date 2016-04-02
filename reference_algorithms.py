import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, expon


#scale factors derived from experimental data, determines relationship
#between distance of command and variance in the direction of the command
variance_scale = 0.43
variance_offset = -0.6

#TODO: create generalized "result object"
#all results from algorithm functions will return a function that has a .pdf method

def estimate_reference_pt(ref, direction):
	center = ref.center
	offset = ref.width/2.*direction if direction[0] else ref.height/2.*direction
	return center + offset

#estimates the "exact" position that a command is referring to 
def estimate_pos(cmd):
	ref = cmd.reference
	direction = cmd.direction
	vector = cmd.distance*direction
	return estimate_reference_pt(ref, direction) + vector

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
def objects_algorithm(cmd, world):
	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]

	# Calculate naive distribution
	naive_dist = naive_algorithm(cmd, world)
	naive_vals = naive_dist.pdf(np.dstack((x, y)))

	# Find Distance to closest object
	ref_dists = {ref : np.sqrt((x - ref.center[0])**2 + (y - ref.center[1])**2) for ref in world.references}
	min_dists = np.min(np.dstack(ref_dists[ref] for ref in ref_dists), axis=2)

	# Difference between distance to closest object and object reference in command
	distance_diff = ref_dists[cmd.reference] - min_dists

	exp_vals = expon.pdf(distance_diff, scale=0.7)

	vals = naive_vals*exp_vals
	loc = np.where(vals == vals.max())
	mean = 0.1*np.array([loc[0][0], loc[1][0]])

	mv_dist = multivariate_normal(mean, naive_dist.cov)

	return mv_dist

def objects_walls_algorithm(cmd, world, k1=4.2, k2=4.4):
	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]

	# Calculate naive distribution
	naive_dist = naive_algorithm(cmd, world)
	naive_vals = naive_dist.pdf(np.dstack((x, y)))

	# Find Distance to closest object
	ref_dists = {ref : np.sqrt((x - ref.center[0])**2 + (y - ref.center[1])**2) for ref in world.references}
	min_ref_dists = np.min(np.dstack(ref_dists[ref] for ref in ref_dists), axis=2)

	# Difference between distance to closest object and object reference in command
	ref_distance_diff = ref_dists[cmd.reference] - min_ref_dists

	ref_distance_vals = expon.pdf(ref_distance_diff, scale=k1)

	# Find distance to nearest wall
	min_wall_dists = np.min(np.dstack((x, y, world.xdim - x, world.ydim - y)), axis=2)

	# Difference between distance to closest wall and object reference in command
	wall_distance_diff = ref_dists[cmd.reference] - min_wall_dists
	wall_distance_diff[wall_distance_diff < 0] = 0

	wall_distance_vals = expon.pdf(wall_distance_diff, scale=k2)

	mean_prob = naive_vals*ref_distance_vals*wall_distance_vals
	loc = np.where(mean_prob == mean_prob.max())
	mean = 0.1*np.array([loc[0][0], loc[1][0]])

	mv_dist = multivariate_normal(mean, naive_dist.cov)

	return mv_dist

# Objects + Walls, closest reference point algorithm
# Uses distance to closest reference pt on object, rather than dist to center
# Significant slowdown
def ow_refpt_algorithm(cmd, world, k1=4.8, k2=3.9):
	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]

	# Calculate naive distribution
	naive_dist = naive_algorithm(cmd, world)
	naive_vals = naive_dist.pdf(np.dstack((x, y)))

	# Find Distance to closest object
	ref_dists = {}
	for ref in world.references:
		directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
		ref_pts = np.array([estimate_reference_pt(ref, direction) for direction in directions])
		possible_dists = np.dstack(np.sqrt((x - pt[0])**2 + (y - pt[1])**2) for pt in ref_pts)
		ref_dists[ref] = np.min(possible_dists, axis=2)
	#ref_dists = {ref : np.sqrt((x - ref.center[0])**2 + (y - ref.center[1])**2) for ref in world.references}
	min_ref_dists = np.min(np.dstack(ref_dists[ref] for ref in ref_dists), axis=2)

	# Difference between distance to closest object and object reference in command
	ref_distance_diff = ref_dists[cmd.reference] - min_ref_dists

	ref_distance_vals = expon.pdf(ref_distance_diff, scale=k1)

	# Find distance to nearest wall
	min_wall_dists = np.min(np.dstack((x, y, world.xdim - x, world.ydim - y)), axis=2)

	# Difference between distance to closest wall and object reference in command
	wall_distance_diff = ref_dists[cmd.reference] - min_wall_dists
	wall_distance_diff[wall_distance_diff < 0] = 0
	#plt.contourf(x, y, wall_distance_diff)
	#plt.show()
	wall_distance_vals = expon.pdf(wall_distance_diff, scale=k2)

	mean_prob = naive_vals*ref_distance_vals*wall_distance_vals
	loc = np.where(mean_prob == mean_prob.max())
	mean = 0.1*np.array([loc[0][0], loc[1][0]])

	mv_dist = multivariate_normal(mean, naive_dist.cov)

	return mv_dist














