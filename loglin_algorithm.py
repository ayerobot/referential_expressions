import numpy as np

class LoglinDistribution:

	def __init__(self, command, world, w=[1, 1, 1]):
		self.command = command
		self.world = world
		self.w = np.matrix(w).T

		x, y = np.mgrid[0:world.xdim:.1, 0:world.ydim:.1]

		self.feature_matrix = get_feature_matrix(x, y, command, world)

		self.normalization = np.sum(np.exp(self.feature_matrix*self.w))

	def pdf(self, pts, normalize=False):
		if pts.ndim == 3:
			x = pts[:,:,0]
			y = pts[:,:,1]
		elif pts.ndim == 2:
			x = pts[:,0]
			y = pts[:,1]
		ref_dists = get_ref_dists(x, y, self.world)

		features = []
		features.append(feature_naive(x, y, self.command, self.world))
		features.append(feature_objects(x, y, self.command, self.world, ref_dists))
		features.append(feature_walls(x, y, self.command, self.world, ref_dists))
		vals = np.exp(np.sum(fi * wi[0] for fi, wi in zip(features, np.array(self.w))))

		if normalize:
			vals = vals/self.normalization
		return vals

	def set_weights(self, w):
		self.w = np.matrix(w).T
		self.normalization = np.sum(np.exp(self.feature_matrix*self.w))

def feature_naive(x, y, cmd, world):
	variance_cmd_parallel = max(cmd.distance*variance_scale + variance_offset, 0.5)
	variance_cmd_ortho = 0.5
	mean = estimate_pos(cmd)
	if cmd.direction[0]:
		varx = variance_cmd_parallel
		vary = variance_cmd_ortho
	else:
		varx = variance_cmd_ortho
		vary = variance_cmd_parallel
	mean_dist = (x - mean[0])**2/varx + (y - mean[1])**2/vary
	return -mean_dist

def get_ref_dists(x, y, world):
	ref_dists = {}
	for ref in world.references:
		directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
		ref_pts = np.array([estimate_reference_pt(ref, direction) for direction in directions])
		possible_dists = np.dstack(np.sqrt((x - pt[0])**2 + (y - pt[1])**2) for pt in ref_pts)
		ref_dists[ref] = np.min(possible_dists, axis=2)
	return ref_dists

def feature_objects(x, y, cmd, world, ref_dists):
	min_ref_dists = np.min(np.dstack(ref_dists[ref] for ref in ref_dists), axis=2)
	ref_distance_diff = ref_dists[cmd.reference] - min_ref_dists
	return -ref_distance_diff

def feature_walls(x, y, cmd, world, ref_dists):
	min_wall_dists = np.min(np.dstack((x, y, world.xdim - x, world.ydim - y)), axis=2)
	wall_distance_diff = ref_dists[cmd.reference] - min_wall_dists
	wall_distance_diff[wall_distance_diff < 0] = 0
	return -wall_distance_diff

def get_feature_matrix(x, y, cmd, world):
	ref_dists = get_ref_dists(x, y, world)

	f_naive = feature_naive(x, y, cmd, world)
	f_objects = feature_objects(x, y, cmd, world, ref_dists)
	f_walls = feature_walls(x, y, cmd, world, ref_dists)

	f_matrix = np.array([el.flatten() for el in (f_naive, f_objects, f_walls)])
	f_matrix = f_matrix.T
	return np.matrix(f_matrix)