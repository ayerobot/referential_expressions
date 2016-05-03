#!/usr/bin/env python
import sys

import numpy as np
import numpy.matlib
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
import nl_parser
from loglin_algorithm import LoglinDistribution, MultiPeakLoglin


"""
Get the data by doing:

	from data_vis import load_data
	data = load_data('point_data.csv')

then data is a dictionary with keys 1-12, referring to the points
"""

#testing functions for finding direction vectors

#index of row with the largest projection on the direction vector
def get_largest_projection(points, direction):
	projections = get_projections(points, direction)
	max_ind = np.argmax(projections, axis=0)
	return max_ind[0]

#gets projections of points on direction
def get_projections(points, direction):
	dir_normalized = np.matrix(direction/np.linalg.norm(direction)).T #transposed for matrix multiplication
	points_mat = np.matrix(points)
	projections = np.array(points_mat.dot(dir_normalized))
	return projections

#this only works on polygons for now, not circles
def farthest_point_direction(direction, obj):
	#the point with the largest projection onto the direction vector
	#max dot product, equivalent to finding the largest element after matrix multiplication
	return obj.position[get_largest_projection(obj.position, direction),:]

#gets the relative reference points means for a rotated object, given a command
#currently (May 2, 2016), the farthest corner point in that direction, 
#and the midpoint of the face with the largest projection in the direction times the length of the face
#TODO: integrate this into a function that returns the reference points for any object
def get_relative_refpts(command):
	direction = command.direction
	obj = command.reference

	point_corner = farthest_point_direction(direction, obj)
	refpoint_corner = point_corner + direction*command.distance

	#determining the face reference point
	#calculating projection of each face's normal vector on the direction
	midpoints, normals, lengths = get_face_info(obj)

	#weight each normal by the length, find the largest projection
	normals_weighted = np.multiply(normals, np.matlib.repmat(lengths, normals.shape[1], 1).T)

	max_proj_ind = get_largest_projection(normals_weighted, direction)

	point_face = midpoints[max_proj_ind, :]

	refpoint_face = point_face + direction*command.distance

	return [refpoint_corner, refpoint_face] #returning array for now, in case this list increases in size

def get_refpt_face(command):
	#determining the face reference point
	#calculating projection of each face's normal vector on the direction
	midpoints, normals, lengths = get_face_info(command.reference)

	#weight each normal by the length, find the largest projection
	normals_weighted = np.multiply(normals, np.matlib.repmat(lengths, normals.shape[1], 1).T)

	max_proj_ind = get_largest_projection(normals_weighted, command.direction)

	point_face = midpoints[max_proj_ind, :]

	#refpoint_face = point_face + command.direction*command.distance

	return point_face


#gets the reference points for an object and corresponding probabilies
#probabilities are hardcoded for now
#TODO: fix terminology
def get_naive_means(command):
	#first determine if object has faces, for now this is if the object is a polygon
	ref = command.reference

	vector = command.direction*command.distance
	if len(ref.position) == 2:
		#this is a circle, the existing estimate_reference_point function should work fine
		#also, only one reference point
		return [estimate_reference_pt(ref, command.direction) + vector], [command.direction], [1.0]
	else:
		#calculate the face refpoint
		refpoint_face = get_refpt_face(command)
		points = []
		points.append(refpoint_face + vector)
		#if there's a corner point that's significantly more in this direction (higher projection), then use corner point
		refpt_face_proj = np.dot(refpoint_face, command.direction/np.linalg.norm(command.direction))

		projections = get_projections(ref.position, command.direction)
		threshold = 2 #inch
		locs = np.where(np.abs(projections - refpt_face_proj) > threshold)
		if len(locs[0]) > 0:
			# max_ind = np.argmax(projections, axis=0)
			# point_corner = ref.position[max_ind[0],:]
			# refpt_corner = point_corner + command.direction*command.distance
			point_corner = farthest_point_direction(command.direction, command.reference)
			refpoint_corner = point_corner + vector
			points.append(refpoint_corner)

		#for now, return an equal probability distribution 
		#direction vectors are also all the same 
		return points, [command.direction for point in points], [1.0/len(points) for point in points]	

#a list of all the face midpoints and normal vectors
def get_face_info(obj):

	points = obj.position

	normals = np.ndarray(obj.position.shape) #same shape
	midpoints = np.ndarray(obj.position.shape) 
	lengths = []
	num_rows = obj.position.shape[0]
	for i in range(0, num_rows): #iterate through rows
		midpoints[i, :] = np.mean(np.array([points[i,:], points[(i+1)%num_rows,:]]), axis=0)
		normals[i, :] = get_normal_vec(points[i,:] - points[(i+1)%num_rows,:])
		lengths.append(np.linalg.norm(points[i,:] - points[(i+1)%num_rows,:]))

	return midpoints, normals, np.array(lengths)

#in the relative reference frame
def estimate_pos_relative(cmd, reference_pt):
	ref = cmd.reference
	direction = cmd.direction
	vector = cmd.distance*direction
	return reference_pt + vector

#returns perpendicular unit vector
def get_normal_vec(vec):
	perp = np.array([-vec[1], vec[0]])
	return perp/np.linalg.norm(perp)

#estimates the pos normal to a reference face. 2x2 vector, each row is a point
#computes the mean of the point, then the normal vector
#scales to unit vector, multiplies by direction
def estimate_pos_normal(cmd, ref_face):
	midpt = np.mean(ref_face, axis=0)
	normal_vec = get_normal_vec(ref_face[0, :] - ref_face[1, :]) #TODO: proper direction! 
	return midpt + normal_vec*cmd.distance

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
		#estimated_pos = estimate_pos(commands[i])
		#plt.scatter(estimated_pos[0], estimated_pos[1], c=np.array([0, 0, 0, 1]), marker='o')
		#plt.text(estimated_pos[0] + 0.25, estimated_pos[1] + 0.25, str(i))

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
def visualize_command(data, world, commands, cmd_num, block = True):
	ax = plt.gca()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	objects = PatchCollection([ref.patch for ref in world.references])
	objects.set_array(np.array([0, 0, 0, 1]))
	ax.add_collection(objects)
	cmd = commands[cmd_num]

	points_x = data[cmd_num][:,0]
	points_y = data[cmd_num][:,1]

	for i in range(0, len(points_x)):
		plt.text(points_x[i], points_y[i], str(i + 1))

	# #TODO: fix this
	# # you can get this out of the commmand directly, I think
	# _, _, cmd_dir_text, obj_name = nl_parser.parse(commands[cmd_num].sentence)

	# face_midpoints = get_face_midpoints(world, obj_name)

	#farthest_point_dir = farthest_point_direction(cmd.direction, cmd.reference)
	#print farthest_point_dir
	#plt.scatter(farthest_point_dir[0], farthest_point_dir[1], c=np.array([0, 0, 0, 1]), marker='o')

	#refpoints_relative = get_relative_refpts(cmd)
	naive_means, directions, probs = get_naive_means(cmd)

	for pt in naive_means:
		plt.scatter(pt[0], pt[1], c='green', marker='o')

	#pos_corner = farthest_point_direction(commands[cmd_num], farthest_point_dir)
	# plt.scatter(pos_corner[0], pos_corner[1], c='green', marker='o')


	# for i in range(0, len(face_midpoints)):
	# 	plt.scatter(face_midpoints[i, 0], face_midpoints[i, 1], c=np.array([0, 0, 0, 1]), marker='o')

	# #calculating naive means with relative reference frame
	# for i in range(0, len(face_midpoints)):
	# 	pos = estimate_pos_relative(commands[cmd_num], face_midpoints[i,:])
	# 	plt.scatter(pos[0], pos[1], c='red', marker='o')


	# obj = [ref for ref in world.references if ref.name == "the keyboard"]
	# obj = obj[0] #convert from array to reference

	# keyboard_points = obj.position #list of points

	# #calculating naive means normal to each face
	# for i in range(0, len(face_midpoints)):
	# 	face = np.array([keyboard_points[i,:], keyboard_points[(i+1)%len(face_midpoints),:]])
	# 	pos = estimate_pos_normal(commands[cmd_num], face)
	# 	plt.scatter(pos[0], pos[1], c='blue', marker='o')


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

#visualizer for all loglin distributions
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
	with open(datafile) as dat:
		data = pickle.load(dat)

		# distribution = algs_to_test['loglin'](commands[11], world)
		visualize_all_multipeak(data, commands, world)

		cmd_num = 1
		cmd = commands[cmd_num]
		cmd_data = data[cmd_num]
		# direction = np.array([-1, 0])
		# direction_2 = np.array([-3.57, -3.28])

		# vec = cmd.distance*direction/np.linalg.norm(direction)
		# vec2 = cmd.distance*direction_2/np.linalg.norm(direction_2)
		# refpt = farthest_point_direction(world, "left of", "the keyboard")
		# refpt_2 = np.array([7.5, 5.4])

		# naive_mean = refpt + vec
		# naive_2 = refpt_2 + vec2


		# directions = [direction, direction, direction_2]
		# naive_means = [naive_mean, np.array([9.19, 10.93]) + vec, naive_2]

		# probs = [0.33, 0.33, 0.33]

		#distribution = LoglinDistribution(commands[11], direction, naive_mean, world)

		#naive_means, directions, probs = get_naive_means(cmd)
		# print naive_means
		# print directions
		# print probs

		#distribution = MultiPeakLoglin(cmd, directions, naive_means, probs, world)
		# naive_mean_old = estimate_pos(cmd)
		# distribution = LoglinDistribution(cmd, cmd.direction, naive_mean_old, world)
		# print cmd.sentence
		# visualize_distribution(cmd_data, distribution, world)

		#visualize_commands(data, world, commands, [1, 2, 3, 4])

	# if len(sys.argv) > 2 and sys.argv[2] != 'save':
	# 	if len(sys.argv) > 3 and sys.argv[3] == 'save':
	# 		print "Saved"
	# 		visualize_all_distributions(data, commands, sys.argv[2], world, sys.argv[4])
	# 	else:
	# 		visualize_all_distributions(data, commands, sys.argv[2], world)
	# elif len(sys.argv) == 4 and sys.argv[2] == 'save':
	# 	print "Saved"
	# 	visualize(data, world, commands, sys.argv[3])
	# else:
	# 	visualize(data, world, commands)