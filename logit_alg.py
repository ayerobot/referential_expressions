#testing the use of logistic regression for this data

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import pickle
from reference_algorithms import *
from world_objects import *
from algorithm_evaluation import generate_feature_vectors
from sklearn.linear_model import LogisticRegression

#rough process for now:
#define features - the features we use already should work
#for each command, use the cheating distribution to generate training data from both phi = 1 (point is valid) and phi = 0 (invalid point for command)
#use the given data for test data? or something?
#generate feature vectors for each training point for each command 
#train the classifier
#test the classifier (on the real data?)

#graph the classifier:
#for each command, generate feature vectors for every point on the grid
#find the probability of each feature vector
#display that on a graph


#returns a 3d array of all feature vectors for a given world and command
#TODO: update this for changing world size
def get_all_features_world(points, command, world):
	#get feature vectors for all points
	feats_all = np.ndarray((480, 360, 3))

	print "calculating feature values for all points"
	for i in range(0, 480):
		for j in range(0, 360):
			point = points[i, j]
			feat_vec = get_feature_vals(point[0], point[1], command, world)
			feats_all[i, j] = feat_vec

	return feats_all

#returns a 2d array of all logit probabilities for the world (i.e the feature matrix)
def probs_cmd_world(logit, feats):
	probs_all = np.ndarray((480, 360))
	for i in range(0, 480):
		for j in range(0, 360):
			features = feats[i, j]
			probs_all[i, j] = logit.predict_proba(features.reshape(1, -1))[0,1] #it doesn't like it if you pass in a 1d array
	return probs_all


#convenience function, thresholds the PDF results and splits the world into two sets of coordinates depending on PDF\
#randomly picks out training and test data according to the training and test fraction
def get_train_test(pdf_results, command, world_coords, train_fraction=0.75):

	#determines the threshold: third of the maximum PDF value
	#almost definitely a better way to do this
	thresh = np.max(pdf_results)/3

	points_1 =  world_coords[pdf_results > thresh]
	points_0 = world_coords[pdf_results < thresh]

	feats_1 = feat_vals_vec(points_1, command, world)
	feats_0 = feat_vals_vec(points_0, command, world)

	all_labels = np.vstack((np.ones((points_1.shape[0], 1)), np.zeros((points_0.shape[0], 1))))

	all_data = np.vstack((feats_1, feats_0))

	num_pts = all_labels.shape[0]

	#create an array of indices, shuffle it around
	#performing random sampling of data 
	inds = np.arange(0, num_pts - 1)
	np.random.shuffle(inds)

	train_fraction = 0.75

	train_end = int(round(train_fraction*num_pts))

	train_data = all_data[0:train_end,:]
	train_labels = all_labels[0:train_end,:]

	test_data = all_data[train_end:,:]
	test_labels = all_labels[train_end:, :]

	return train_data, train_labels, test_data, test_labels

#trains a logit function on a particular command (primarily for testing)
def train_logit_cmd(data, cmd_num, commands, world, world_coords):
	command = commands[cmd_num]
	#will use the cheating distribution to generate training and test data
	#this is probably the biggest weakness here, as this won't generalize to the rotated objects without
	#a new distribution being used
	gauss = cheating_algorithm(data[cmd_num]) 

	pdf_results = gauss.pdf(world_coords)

	train_data, train_labels, test_data, test_labels = get_train_test(pdf_results, command, world_coords)
	
	logit = LogisticRegression()

	print "training logistic regression"
	logit.fit(train_data, train_labels.reshape((-1,)))

	print "feature weights: " + str(logit.coef_)

	print "testing on test data"

	predictions = logit.predict(test_data)

	print "accuracy: " + str(sum(np.equal(predictions, test_labels.reshape((-1,))))/(float(test_labels.shape[0])))
	return logit

#trains a logistic classifier on all data for a particular world
def train_logit_all_world(data, commands, world, world_coords):
	train_data_list = []
	train_label_list = []
	test_data_list = []
	test_label_list = []
	#loops through all commands, producing train and test data for each
	#appends them to the list
	#could also just create all data and then shuffle them separately
	for i in range(1, len(data.keys()) + 1): 
		command = commands[i]
		gauss = cheating_algorithm(data[cmd_num]) 
		pdf_results = gauss.pdf(world_coords)
		train_data, train_labels, test_data, test_labels = get_train_test(pdf_results, command, world_coords)
		#probably a more elegant way to do this
		train_data_list.append(train_data)
		train_label_list.append(train_labels)
		test_data_list.append(test_data)
		test_label_list.append(test_labels)

	#place all training and test data together:
	train_data_all = np.vstack(train_data_list)
	train_labels_all = np.vstack(train_label_list)
	test_data_all = np.vstack(test_data_list)
	test_labels_all = np.vstack(test_label_list)

	#creates logit object, trains, tests

	logit = LogisticRegression()

	print "training logistic regression"
	logit.fit(train_data_all, train_labels_all.reshape((-1,))) #need to reshape to be 1d array

	print "feature weights: " + str(logit.coef_)

	print "testing on test data"

	predictions = logit.predict(test_data_all)

	print "accuracy: " + str(sum(np.equal(predictions, test_labels_all.reshape((-1,))))/(float(test_labels_all.shape[0])))
	return logit


#derived from visualize_all_distributions THX ROSHAN
def plot_all_commands(logit,data,  commands, world, world_coords):
	fig = plt.figure(figsize=(18, 8))
	fig.subplots_adjust(hspace=0.5)
	for i in range(1, 13):
		plt.subplot(3, 4, i)
		plot_logit(logit, data[i], i, world, world_coords, block=False)
		plt.title(commands[i].sentence, fontsize=12)

	plt.show()

#plots a logistic function on top of the data points
def plot_logit(logit, points, cmd_num, world, world_coords, block=True):
	#calculate probability of each point (un-normalized)
	#probably best to precompute this
	print "generating probabilities for all feature vectors"
	feats_all = get_all_features_world(world_coords, commands[cmd_num], world)
	probs_all = probs_cmd_world(logit, feats_all)
	
	ax = plt.gca()
	ax.set_xlim([0, world.xdim]) # Set x dim to 4 feet
	ax.set_ylim([0, world.ydim]) # Set y dim to 3 feet

	plt.contourf(x, y, probs_all)

	objects = PatchCollection([ref.patch for ref in world.references], cmap=cm.gray, alpha=1)
	objects.set_array(np.array([1, 1, 1]))
	ax.add_collection(objects)

	plt.scatter(points[:, 0], points[:, 1], c=np.array([1, 1, 1, 1]), marker='.')

	if block:
		plt.show()


if __name__ == "__main__":

	#loading data:

	scenenum = 1
	datafile = 'data/scene_' + str(scenenum) + '_images_annotated_preprocessed.dat'
	scenenum = int(scenenum)
	commands = all_commands[scenenum]
	world = all_worlds[scenenum]
	with open(datafile) as dat:
		data = pickle.load(dat)

	#testing on the first command:

	cmd_num = 1
	x, y = np.mgrid[0:world.xdim:.01, 0:world.ydim:.01]
	stacked = np.dstack((x, y))

	#logit = train_logit_cmd(data, cmd_num, commands, world, stacked)
	logit = train_logit_all_world(data, commands, world, stacked)
	#plot_logit(logit, data[cmd_num], cmd_num, world, stacked)
	plot_all_commands(logit, data, commands, world, stacked)
	
	









	# 