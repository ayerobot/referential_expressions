#!/usr/bin/env python
#determines x/y pixel coordinates of points on the image using matplotlib and handler
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

x_len = 48 #inches
y_len = 36 #inches

#handles the annotations 
class Annotator:
	def __init__(self):
		self.points = [] #list containing annotated points in order: bottom left corner, bottom right corner, top left corner, then points 1-12
	def onclick(self, event):
		#TODO: draw points on screen to show what you've annotated
		if len(self.points) < 16:
			tup = (event.xdata, event.ydata)
			print str(tup)
			self.points.append(tup)
	def reset(self, event):
		self.points = []
	def save(self, event): #closes the plot window
		if len(self.points) < 15:
			print "I think you forgot some points"
		else:
			plt.close()

#TODO: probably a more abstracted function for performing this conversion, to prevent code-reuse
class General_Annotator:
	def __init__(self):
		self.basis_pts = [] #list containing basis pts: BL corner, BR corner, TL corner
	def onclick(self, event):
		tup = (event.xdata, event.ydata)
		if len(self.basis_pts) < 3:
			print "calibration point"
			self.basis_pts.append(tup)
			#finalizing if basis vectors are here
			if len(self.basis_pts) == 3:
				print "all basis vectors gathered!"
				bottom_left = np.matrix(self.basis_pts[0]) #will be size (1, 2), need to transpose later
				bottom_right = np.matrix(self.basis_pts[1])
				top_left = np.matrix(self.basis_pts[2])

				#new basis vectors
				b1 = bottom_right - bottom_left #x basis
				b2 = top_left - bottom_left #y basis

				self.basis_vecs = np.concatenate((b1.T, b2.T), axis=1)
		else: 
			print "converting point"
			print self.convert_point(tup)
	def convert_point(self, point):
		#assuming you've already defined the basis vectors
		point_subtracted = np.matrix(point) - np.matrix(self.basis_pts[0])
		point_new_basis = np.multiply(np.linalg.inv(self.basis_vecs)*(point_subtracted.T), np.matrix([x_len, y_len]).T)
		return point_new_basis


def annotate_image(imgpath):
	img = mpimg.imread(imgpath)
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	plt.title(imgpath)
	imgplot = ax.imshow(img)

	img_annotator = Annotator()

	axsave = plt.axes([0.9, 0.05, 0.05, 0.075])
	axreset = plt.axes([0.9, 0.01, 0.05, 0.05])
	savebutton = Button(axsave, 'Save')
	savebutton.on_clicked(img_annotator.save)
	resetbutton = Button(axreset, 'Reset')
	resetbutton.on_clicked(img_annotator.reset)

	#connect event handler
	fig.canvas.mpl_connect('button_press_event', img_annotator.onclick)
	plt.show()

	return img_annotator.points[0:15] #it counts the last click to the save button as a click


#after the reference points are picked, converts any future point into the correct coordinate frame
def general_annotator(imgpath):
	img = mpimg.imread(imgpath)
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	plt.title(imgpath)
	imgplot = ax.imshow(img)

	print "pick three points for calibration: BL corner, BR corner, TL corner"

	img_annotator = General_Annotator()
	#connect event handler
	fig.canvas.mpl_connect('button_press_event', img_annotator.onclick)
	plt.show()


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "image_annotator.py imagedirectory"
	else:		
		print "opening all images in " + sys.argv[1]
		imglist = [sys.argv[1] + "/" + item for item in os.listdir(sys.argv[1]) if item[-4:] == ".JPG"] #getting all JPG files
		results = [] #list of lists
		for img in imglist:
			result = annotate_image(img)
			results.append(result)

		#saving as numpy file, will process later
		np.save(sys.argv[1] + "_annotated", results)



