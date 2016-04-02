#determines x/y pixel coordinates of points on the image using matplotlib and handler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import os
import sys

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



