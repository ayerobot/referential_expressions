""" Locational RefExes : PARSER """

import sys
import re
import string



# ASSUME NO NUMBERS GREATER THAN NINETY NINE
units = {"A" : 1, "a" : 1, "an" : 1, "An" : 1, "one" : 1, 
			"two" : 2, "three" : 3, "four" : 4, "five" : 5, 
			"six" : 6, "seven" : 7, "eight" : 8, "nine" : 9, "ten" : 10, 
			"eleven" : 11, "twelve" : 12, "thirteen" : 13, "fourteen" : 14, 
			"fifteen" : 15, "sixteen" : 16, "seventeen" : 17,
			"eighteen" : 18, "nineteen" : 19}

tens = {"twenty" : 20, "thirty" : 30, "forty" : 40, "fifty" : 50, "sixty" : 60, "seventy" : 70, "eighty" : 80, "ninety" : 90}

fractions = {"quarter" : .25, "half" : .5, "quarters" : .75}


def parseNumber(words):
	# length of word
	l = 0
	# index of word
	i = 0
	if words[i] in tens:
		N = tens[words[i]]
		l += len(words[i]) + 1
		i += 1
		if words[i] in units:
			N += units[words[i]]
			l += len(words[i]) + 1
			i += 1
	elif words[i] in units:
		N = units[words[i]]
		l += len(words[i]) + 1
		i += 1
	else:
		try:
			N = float(words[i]) # throw some kind of exception if not a float
			l += len(words[i]) + 1
			i +=1
		except ValueError:
			print (words[i] + " is not a valid number.")
			return None, 0

	if words[i] == "and":
		l += 4
		if words[i+2] in fractions:
			N += fractions[words[i+2]]
			l += len(words[i+1]) + 1 + len(words[i+2]) + 1
			i += 3

	return N, i, l


measurement = {"inch" : 1, "inches" : 1, "foot" : 12, "feet" : 12}
relations = ["to the left of", "to the right of", "left of", "right of", "in front of", "behind"]
relationsRegEx = "to the left of|to the right of|left of|right of|in front of|behind"

def parse(command):
	""" parses command, outputs

	N: float, numerical information, in inches
	M: string containing the standard of measurement used in the command
	R: string, relation, the direction in which point is located
	O: string, object "landmark", in reference to which point is located


	"""
	words = re.split(r'\s|-', command) # split by whitespace and dashes
	
	if (len(words) < 5): # minimum length of words in the command
		print("command is not in full form")
		return

	N, i, l = parseNumber(words)

	if N == None:
		return

	if words[i] in measurement:
		M = words[i]
		N *= measurement[words[i]]
		i += 1
	else: 
		print (words[i] + " is not a measurement we handle.")
		return

	command = command[l + len(M) + 1:]
	rel = re.match(relationsRegEx,command)
	if (rel == None):
		print "No relations 'left of' 'right of' 'in front of' or 'behind' in command."
		return

	R = rel.group(0)

	O = command[len(R) + 1:]

	return N, M, R, O


if __name__ == '__main__':

	command = sys.argv[1]
	print parse(command)
