import numpy as np
from matplotlib.patches import Circle, Polygon

from nl_parser import parse

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

#SCENE 1 

# Object dimensions and commands from https://docs.google.com/document/d/1TbAKCrdEfgD6nCEjhlpJ4BpAJcmeSajsGQwb4HBsh7U/edit
#TODO: relationship between "the" and object name
keyboard = Reference("the keyboard", np.array([[33.375, 13.5], [38.5, 13.5], [38, 24.625], [33, 24.375]]))
car = Reference("the car", np.array([[9.5, 21.5], [10.625, 21.25], [11.625, 24], [10.375, 24.5]]))
bowl = Reference("the pink bowl", (np.array([19.75, 10]), 2.5))

world = World([keyboard, car, bowl], 48, 36)

commands = {
	1 : Command("4 inches left of the car", world),
	2 : Command("1 foot left of the keyboard", world),
	3 : Command("1 and a half feet right of the car",world),
	4 : Command("4 inches in front of the pink bowl", world),
	5 : Command("5 and a half inches behind the keyboard", world),
	6 : Command("2 inches left of the pink bowl", world),
	7 : Command("1 and a half inches behind the car", world),
	8 : Command("7 inches right of the pink bowl", world ),
	9 : Command("16 inches in front of the car", world),
	10 : Command("2 feet behind the pink bowl",world),
	11 : Command("3 inches right of the keyboard",world),
	12 : Command("4 and a half inches in front of the keyboard",world)
}

#SCENE 2

