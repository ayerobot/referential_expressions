#!/usr/bin/env python

import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys

"""
Loads data from points_data.csv
"""
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

if __name__ == '__main__':
	fig, ax = plt.subplots()
	ax.set_xlim([0, 48]) # Set x dim to 4 feet
	ax.set_ylim([0, 36]) # Set y dim to 3 feet

	# Object dimensions from https://docs.google.com/document/d/1TbAKCrdEfgD6nCEjhlpJ4BpAJcmeSajsGQwb4HBsh7U/edit
	keyboard = Polygon([[33.375, 13.5], [38.5, 13.5], [38, 24.625], [33, 24.375]], True)
	car = Polygon([[9.5, 21.5], [10.625, 21.25], [11.625, 24], [10.375, 24.5]], True)
	bowl = Circle((19.75, 10), 2.5)

	objects = PatchCollection([keyboard, car, bowl])
	ax.add_collection(objects)

	data = load_data('point_data.csv')

	colors = cm.rainbow(np.linspace(0, 1, 12))
	for i in range(1, 13):
		plt.scatter(data[i][:,0], data[i][:,1], c=colors[i-1])

	if len(sys.argv) == 3 and sys.argv[1] == 'save':
		plt.savefig(sys.argv[2], format='pdf')

	plt.show()