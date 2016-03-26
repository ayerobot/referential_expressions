#converting data from pixels on a photograph to inches relative to the whiteboard
#essentially, relies on measuring the corners of the board and computing a change of basis relative to those vectors
#so the locations of the data points relative to the board can be determined

import numpy as np
import csv
import pickle

x_len = 48 #inches
y_len = 36 #inches
y_pix_total = 1907 #used to reorient y coordinate, move the origin to the lower left corner
num_setups = 4
num_trials = 12


#converts a tuple in text file format "x, y"
def string_to_vec(s):
	return np.matrix([int(elem.strip()) for elem in s.split(",")]).T

def preprocess_data_csv(filename):
	with open(filename) as csvfile:
		csvreader = csv.reader(csvfile)

		results = {}
		for i in range(1, num_setups + 1):
			#each key will be a setup
			results[i] = {} #will contain a dictionary with keys equal to the trial number, to group points together

		for row in csvreader:
			#ignore the first row:
			if row[0] != 'Setup': 

				#first element is the setup, second is bottom left (reference) corner, third is bottom right corner, fourth is top left corner
				#next elements are the results for each command (1 through whatever)

				setup_num = int(row[0])

				#getting the basis change matrix
				#could probably do this in a loop but this is clearer
				bottom_left = string_to_vec(row[1])
				bottom_right = string_to_vec(row[2])
				top_left = string_to_vec(row[3])
				#new basis vectors
				b1 = bottom_right - bottom_left #x basis
				b2 = top_left - bottom_left #y basis

				basis_vecs = np.concatenate((b1, b2), axis=1)

				#computing scale, will scale points at the end of the transformation

				for i in range(4, len(row)): #for every elem
					point = string_to_vec(row[i]) - bottom_left
					#point = point.reshape(len(point), 1) #performing transposition on 1d array
					print np.linalg.inv(basis_vecs)*(point)
					point_new_basis = np.multiply(np.linalg.inv(basis_vecs)*(point), np.matrix([x_len, y_len]).T)
					results[setup_num][i - 3] = results[setup_num].get(i - 3, []) + [np.array(point_new_basis)] #casting back into array

	#save as pickle
	with open("outfile.dat", "wb") as outfile:
		pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
	filename = "exp_2_test.csv"
	preprocess_data(filename)