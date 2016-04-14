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
keyboard_1 = Reference("the keyboard", np.array([[33.375, 13.5], [38.5, 13.5], [38, 24.625], [33, 24.375]]))
car_1 = Reference("the car", np.array([[9.5, 21.5], [10.625, 21.25], [11.625, 24], [10.375, 24.5]]))
bowl_1 = Reference("the pink bowl", (np.array([19.75, 10]), 2.5))

world_1 = World([keyboard_1, car_1, bowl_1], 48, 36)

commands_1 = {
	1 : Command("4 inches left of the car", world_1),
	2 : Command("1 foot left of the keyboard", world_1),
	3 : Command("1 and a half feet right of the car",world_1),
	4 : Command("4 inches in front of the pink bowl", world_1),
	5 : Command("5 and a half inches behind the keyboard", world_1),
	6 : Command("2 inches left of the pink bowl", world_1),
	7 : Command("1 and a half inches behind the car", world_1),
	8 : Command("7 inches right of the pink bowl", world_1 ),
	9 : Command("16 inches in front of the car", world_1),
	10 : Command("2 feet behind the pink bowl",world_1),
	11 : Command("3 inches right of the keyboard",world_1),
	12 : Command("4 and a half inches in front of the keyboard",world_1)
}

#SCENE 2

keyboard_2 = Reference("the keyboard", np.array([[26.83068791, 19.04820551], [37.72432027, 18.35193125], [37.91045218, 23.30818498], [26.80185057, 23.8928784 ]]))
car_2 = Reference("the car", np.array([[17.99056582, 10.03777581], [19.06912902, 9.70684443], [ 19.00192333, 7.08450112], [ 17.92537379, 7.0771387]]))
bowl_2 = Reference("the pink bowl", (np.array([15.41, 25.58]), 2.5))

world_2 = World([keyboard_2, car_2, bowl_2], 48, 36)

commands_2 = {
	1 : Command("2 inches left of the car", world_2),
	2 : Command("1 and a quarter feet left of the keyboard", world_2),
	3 : Command("6 inches in front of the pink bowl",world_2),
	4 : Command("3 and a half inches behind the keyboard", world_2),
	5 : Command("10 inches left of the pink bowl", world_2),
	6 : Command("1 and a half feet behind the car", world_2),
	7 : Command("1 inch right of the pink bowl", world_2),
	8 : Command("1 foot right of the car", world_2),
	9 : Command("4 inches in front of the car", world_2),
	10 : Command("5 and a half inches behind the pink bowl",world_2),
	11 : Command("7 inches right of the keyboard",world_2),
	12 : Command("1 and a half inches in front of the keyboard",world_2)
}


#SCENE 3

keyboard_3 = Reference("the keyboard", np.array([[9.42, 4.27], [17.26, 12.31], [13.2, 15.75], [5.61, 7.32]]))
car_3 = Reference("the car", np.array([[35.52, 10.5], [36.24, 10.5], [36.5, 13.88],  [35.5, 13.8]]))
bowl_3 = Reference("the pink bowl", (np.array([21.52, 27.12]), 2.5))

world_3 = World([keyboard_3, car_3, bowl_3], 48, 36)

commands_3 = {
	1 : Command("1 and a half feet left of the car", world_2),
	2 : Command("2 feet right of the keyboard", world_2),
	3 : Command("7 inches right of the car",world_2),
	4 : Command("8 inches in front of the pink bowl", world_2),
	5 : Command("12 inches behind the keyboard", world_2),
	6 : Command("1 foot left of the pink bowl", world_2),
	7 : Command("13 inches behind the car", world_2),
	8 : Command("4 inches right of the pink bowl", world_2),
	9 : Command("2 and a half inches in front of the car", world_2),
	10 : Command("1 foot behind the pink bowl",world_2),
	11 : Command("3 inches left of the keyboard",world_2),
	12 : Command("1 inch in front of the keyboard",world_2)
}

all_worlds = [world_1, world_2, world_3]
all_commands = [commands_1, commands_2, commands_3]


