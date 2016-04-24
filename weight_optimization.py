import numpy as np
import scipy.io as sio

from reference_algorithms import *

def gridsearch(data, commands, world):
	k1 = np.arange(0.5, 0.7, .01)
	k2 = np.arange(0.1, 0.25, .01)
	k3 = np.arange(0.2, 0.3, .01)
	all_probs = np.zeros((len(k1), len(k2), len(k3)))
	#cheating_prob = get_cheating_prob(data)
	loglin_dists = [loglin_alg(commands[num], world) for num in range(1, 13)]
	for i, val1 in enumerate(k1):
		for j, val2 in enumerate(k2):
			for k, val3 in enumerate(k3):
				print float(len(k2)*len(k3)*i + len(k3)*j + k)/all_probs.size*100, "%"
				for dist in loglin_dists:
					dist.set_weights([val1, val2, val3])
				#probs = np.array([np.sum(np.log(loglin_alg(commands[num], world, [val1, val2, val3]).pdf(data[num], normalize=True))) for num in range(1, 13)])
				probs = [dist.pdf(data[num], normalize=True) for dist, num in zip(loglin_dists, range(1, 13))]
				logsum = np.sum(np.log(probs))
				all_probs[i, j, k] = logsum
	print "100.0 %"
	loc = np.where(all_probs == all_probs.max())
	print
	print 'k1 =', k1[loc[0][0]]
	print 'k2 =', k2[loc[1][0]]
	print 'k3 =', k3[loc[2][0]]
	return all_probs


def generate_feature_vectors(data, commands, world):
	x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]
	for i in range(1, 13):
		Tx = get_feature_matrix(x, y, commands[i], world)
		theta = np.mean(get_feature_matrix(data[i][:,0], data[i][:,1], commands[i], world), 0)
		sio.savemat('grad_descent/command' + str(i) + '.mat', {'Tx' : Tx, 'theta' : theta})