import numpy as np

#IN DEVELOPMENT
#functions for finding reference points from objects, useful for rotated objects

#testing functions for finding direction vectors

#index of row with the largest projection on the direction vector
def get_largest_projection(points, direction):
	projections = get_projections(points, direction)
	max_ind = np.argmax(projections, axis=0)
	return max_ind[0]

#gets projections of points on direction
def get_projections(points, direction):
	dir_normalized = np.matrix(direction/np.linalg.norm(direction)).T #transposed for matrix multiplication
	points_mat = np.matrix(points)
	projections = np.array(points_mat.dot(dir_normalized))
	return projections

#this only works on polygons for now, not circles
def farthest_point_direction(direction, obj):
	#the point with the largest projection onto the direction vector
	#max dot product, equivalent to finding the largest element after matrix multiplication
	return obj.position[get_largest_projection(obj.position, direction),:]

#gets the relative reference points means for a rotated object, given a command
#currently (May 2, 2016), the farthest corner point in that direction, 
#and the midpoint of the face with the largest projection in the direction times the length of the face
#TODO: integrate this into a function that returns the reference points for any object
def get_relative_refpts(command):
	direction = command.direction
	obj = command.reference

	point_corner = farthest_point_direction(direction, obj)
	refpoint_corner = point_corner + direction*command.distance

	#determining the face reference point
	#calculating projection of each face's normal vector on the direction
	midpoints, normals, lengths = get_face_info(obj)

	#weight each normal by the length, find the largest projection
	normals_weighted = np.multiply(normals, np.matlib.repmat(lengths, normals.shape[1], 1).T)

	max_proj_ind = get_largest_projection(normals_weighted, direction)

	point_face = midpoints[max_proj_ind, :]

	refpoint_face = point_face + direction*command.distance

	return [refpoint_corner, refpoint_face] #returning array for now, in case this list increases in size

def get_refpt_face(command):
	#determining the face reference point
	#calculating projection of each face's normal vector on the direction
	midpoints, normals, lengths = get_face_info(command.reference)

	#weight each normal by the length, find the largest projection
	normals_weighted = np.multiply(normals, np.matlib.repmat(lengths, normals.shape[1], 1).T)

	max_proj_ind = get_largest_projection(normals_weighted, command.direction)

	point_face = midpoints[max_proj_ind, :]

	#refpoint_face = point_face + command.direction*command.distance

	return point_face


#gets the reference points for an object and corresponding probabilies
#probabilities are hardcoded for now
#TODO: fix terminology
def get_naive_means(command):
	#first determine if object has faces, for now this is if the object is a polygon
	ref = command.reference

	vector = command.direction*command.distance
	if len(ref.position) == 2:
		#this is a circle, the existing estimate_reference_point function should work fine
		#also, only one reference point
		return [estimate_reference_pt(ref, command.direction) + vector], [command.direction], [1.0]
	else:
		#calculate the face refpoint
		refpoint_face = get_refpt_face(command)
		points = []
		points.append(refpoint_face + vector)
		#if there's a corner point that's significantly more in this direction (higher projection), then use corner point
		refpt_face_proj = np.dot(refpoint_face, command.direction/np.linalg.norm(command.direction))

		projections = get_projections(ref.position, command.direction)
		threshold = 2 #inch
		locs = np.where(np.abs(projections - refpt_face_proj) > threshold)
		if len(locs[0]) > 0:
			# max_ind = np.argmax(projections, axis=0)
			# point_corner = ref.position[max_ind[0],:]
			# refpt_corner = point_corner + command.direction*command.distance
			point_corner = farthest_point_direction(command.direction, command.reference)
			refpoint_corner = point_corner + vector
			points.append(refpoint_corner)

		#for now, return an equal probability distribution 
		#direction vectors are also all the same 
		return points, [command.direction for point in points], [1.0/len(points) for point in points]	

#a list of all the face midpoints and normal vectors
def get_face_info(obj):

	points = obj.position

	normals = np.ndarray(obj.position.shape) #same shape
	midpoints = np.ndarray(obj.position.shape) 
	lengths = []
	num_rows = obj.position.shape[0]
	for i in range(0, num_rows): #iterate through rows
		midpoints[i, :] = np.mean(np.array([points[i,:], points[(i+1)%num_rows,:]]), axis=0)
		normals[i, :] = get_normal_vec(points[i,:] - points[(i+1)%num_rows,:])
		lengths.append(np.linalg.norm(points[i,:] - points[(i+1)%num_rows,:]))

	return midpoints, normals, np.array(lengths)

#in the relative reference frame
def estimate_pos_relative(cmd, reference_pt):
	ref = cmd.reference
	direction = cmd.direction
	vector = cmd.distance*direction
	return reference_pt + vector

#returns perpendicular unit vector
def get_normal_vec(vec):
	perp = np.array([-vec[1], vec[0]])
	return perp/np.linalg.norm(perp)

#estimates the pos normal to a reference face. 2x2 vector, each row is a point
#computes the mean of the point, then the normal vector
#scales to unit vector, multiplies by direction
def estimate_pos_normal(cmd, ref_face):
	midpt = np.mean(ref_face, axis=0)
	normal_vec = get_normal_vec(ref_face[0, :] - ref_face[1, :]) #TODO: proper direction! 
	return midpt + normal_vec*cmd.distance