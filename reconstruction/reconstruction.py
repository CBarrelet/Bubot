import cv2
import numpy as np
import torch.nn as nn
from torch import tensor
from scipy.optimize import linear_sum_assignment

class Measurement(object):

    def __init__(self, center=np.array([0,0,0], dtype=float), height=float(0), width=float(0)):
        self.center = center
        self.height = height
        self.width = width

def cosinus_distance(vec_1, vec_2):
	"""
	Compute the cosinus distance between two feature vectors.

	Parameters:
		- vec_1: torch.tensor
			A feature vector.
		- vec_2: torch.tensor
			A feature vector.
	Return:
		- cos_sim: float()
			The cosinus similarity between those two feature vector.
	"""
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	cos_sim = cos(vec_1, vec_2)
	return cos_sim


def pairing(scores_array):
	"""
	Solve the maximum linear sum assignment problem.
	Reject row/column pairs where the sum is equal to 0. 

	Parameters:
		- scores_array: ndarray
			A NxM dimensional cost matrix
	Return:
		- paired_list: List[List[int(row_idx), int(col_idx)]
	"""  
	paired_list = []
	row_idx, col_idx = linear_sum_assignment(scores_array, maximize=True)
	paired_list = [[r, col_idx[i]] for i, r in enumerate(row_idx) if scores_array[r, col_idx[i]] != 0]
	return paired_list


def stereo_matching(detections, thres=0.4):
	"""
	Solve left-to-right boxes matching problem.

	Use the epipolar constraint to first find potential matches.
	For each left image detections center y_coordinate y and a delta (height/2),
	find right image detections where their central y_coordinate y' ∈ [y-delta ; y+delta].

	Create a cost matrix according to the threshold and perform linear assignment. 
	Update detections with their corresponding matched detection.

	Parameters:
		- detections: List[List[detection], List[detection]]
			A list containing a list of left detections and a list of right detections
		- tresh: float()
			A similarity treshold between two bounding boxes.
			If the similarity is greater than the treshold,
			the similarity is added up to the matching cost matrix.
	"""    
	sim_scores = np.zeros((len(detections[0]), len(detections[1])))
	for i, detection0 in enumerate(detections[0]):
		bbx0 = detection0.xyxy
		center0_y = int((bbx0[1] + bbx0[3])/2)
		delta = int(abs(bbx0[3]-bbx0[1])/4)
		for j, detection1 in enumerate(detections[1]):
			bbx1 = detection1.xyxy
			center1_y = int((bbx1[1] + bbx1[3])/2)
			if(center1_y in range(int(center0_y-delta), int(center0_y+delta))):
				cos_sim = cosinus_distance(tensor(detection0.feature), tensor(detection1.feature))
				if(cos_sim > thres):
					sim_scores[i,j] = cos_sim
	paired_list = pairing(sim_scores)
	#[detection.set_feature_to_cpu() for detection in detections[0]]
	#[detection.set_feature_to_cpu() for detection in detections[1]]
	paired_colors = [[np.random.randint(0, 150) for _ in range(3)] for _ in range(len(paired_list))]
	for i, pair in enumerate(paired_list):
		detections[0][pair[0]].set_stereo(detections[1][pair[1]], paired_colors[i])
		detections[1][pair[1]].set_stereo(detections[0][pair[0]], paired_colors[i])


def triangulate(dataset, point_A, point_B):
	"""
	Triangulate two points according to the dataset calibration file.

	Parameters:
		- dataset: LoadStereoImages
			The current dataset.
		- point_A: List(int(x), int(y))
			A 2D point from the left image.
		- point_B: List(int(x), int(y))
			A 2D point from the right image.
	Return:
		- point_3D: ndarray
			An array containing the 'x', 'y', and 'z' coordinates in the camera space.
	""" 
	P1 = dataset.stereo_params['P1']
	P2 = dataset.stereo_params['P2']
	point_A = (int(point_A[0]), int(point_A[1]))
	point_B = (int(point_B[0]), int(point_B[1]))
	point_4D = cv2.triangulatePoints(P1, P2, point_A, point_B)
	point_3D = np.array([point_4D[0], point_4D[1], point_4D[2]])/ point_4D[3] * dataset.stereo_params['square_size']
	point_3D = np.array([point_3D[0][0], point_3D[1][0], point_3D[2][0]])
	return point_3D

def find_pair(p1_a, p2_a, xyxy_a, xyxy_b, xmax_b, bw_b):
	"""
	Find endpoints from a to b within a 5 pixels delta.
	If no endpoint found, set points to (-1, -1)

	Parameters:
		- p1_a: List(int(x), int(y))
			First endpoint found with PCA on segmentation a
		- p2_a: List(int(x), int(y))
			Second endpoint found with PCA on segmentation a
		- xyxy_a: Detection.xyxy
			Bbx coordinates of the detection a
		- xyxy_b: Detection.xyxy
			Bbx coordinates of the detection b
		- xmax_b: int()
			Width of the segmentation b
		- bw_b: ndarray
			The mask of the segmentation b
	Return:
		p1_ab: List(int(x), int(y))
			First paired point from a to b
		p2_ab: List(int(x), int(y))
			Second paired point from a to b
	"""

	# TODO: Fix try catch "out of image boundaries"

	p1_ab = (-1, -1)
	p2_ab = (-1, -1)

	delta = 5
	i = 0
	try:
		while(i<delta):
			if(p1_a[1]-xyxy_a[1] < p2_a[1]-xyxy_a[1]):
				ext = [(x+xyxy_b[0], p1_a[1]+i) for x in range(xmax_b) if bw_b[p1_a[1]-xyxy_a[1]+i, x] == 255]
			else:
				ext = [(x+xyxy_b[0], p1_a[1]-i) for x in range(xmax_b) if bw_b[p1_a[1]-xyxy_a[1]-i, x] == 255]
			i += 1 if not(ext) else delta
		if(ext):
			p1_ab = ext[0]
			i = 0
			while(i<delta):
				if(p2_a[1]-xyxy_a[1] < p1_a[1]-xyxy_a[1]):
					ext = [(x+xyxy_b[0], p2_a[1]+i) for x in range(xmax_b) if bw_b[p2_a[1]-xyxy_a[1]+i, x] == 255]
				else:
					ext = [(x+xyxy_b[0], p2_a[1]-i) for x in range(xmax_b) if bw_b[p2_a[1]-xyxy_a[1]-i, x] == 255]
				i += 1 if not(ext) else delta
			if(ext):
				p2_ab = ext[-1]
			else:
				p1_ab = (-1, -1)
				p2_ab = (-1, -1)
		else:
			p1_ab = (-1, -1)
			p2_ab = (-1, -1)
	except:
		pass
	return p1_ab, p2_ab


def set_pair_points(detection):
	segmentation_1 = detection.segmentation
	segmentation_2 = detection.stereo_detection.segmentation
	_, bw_1 = cv2.threshold(segmentation_1, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	_, bw_2 = cv2.threshold(segmentation_2, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	ymax_1, xmax_1 = segmentation_1.shape[0:2]
	ymax_2, xmax_2 = segmentation_2.shape[0:2]

	xyxy_1 = detection.xyxy
	xyxy_2 = detection.stereo_detection.xyxy

	p1_width_1 = detection.p1_width
	p2_width_1 = detection.p2_width
	p1_height_1 = detection.p1_height
	p2_height_1 = detection.p2_height

	p1_width_2 = detection.stereo_detection.p1_width
	p2_width_2 = detection.stereo_detection.p2_width
	p1_height_2 = detection.stereo_detection.p1_height
	p2_height_2 = detection.stereo_detection.p2_height
	
	# Find endpoints to measure the width
	# From 1 to 2
	if(p1_width_1 != (-1,-1)) and (p2_width_1 != (-1,-1)):
		p1_width_12, p2_width_12 = find_pair(p1_width_1, p2_width_1, xyxy_1, xyxy_2, xmax_2, bw_2)
	else:
		p1_width_12, p2_width_12 = (-1,-1), (-1,-1)
	# From 2 to 1
	if(p1_width_2 != (-1,-1)) and (p2_width_2 != (-1,-1)):
		p1_width_21, p2_width_21 = find_pair(p1_width_2, p2_width_2, xyxy_2, xyxy_1, xmax_1, bw_1)
	else: 
		p1_width_21, p2_width_21 = (-1,-1), (-1,-1)
	# Find endpoints to measure the height
	# From 1 to 2
	if(p1_height_1 != (-1,-1)) and (p2_height_1 != (-1,-1)):
		p1_height_12, p2_height_12 = find_pair(p1_height_1, p2_height_1, xyxy_1, xyxy_2, xmax_2, bw_2)
	else:
		p1_height_12, p2_height_12 = (-1,-1), (-1,-1)
	# From 2 to 1
	if(p1_height_2 != (-1,-1)) and (p2_height_2 != (-1,-1)):
		p1_height_21, p2_height_21 = find_pair(p1_height_2, p2_height_2, xyxy_2, xyxy_1, xmax_1, bw_1)
	else:
		p1_height_21, p2_height_21 = (-1,-1), (-1,-1)

	detection.p1_width_12 = p1_width_12
	detection.p2_width_12 = p2_width_12
	detection.p1_height_12 = p1_height_12
	detection.p2_height_12 = p2_height_12

	detection.stereo_detection.p1_width_12 = p1_width_21
	detection.stereo_detection.p2_width_12 = p2_width_21
	detection.stereo_detection.p1_height_12 = p1_height_21
	detection.stereo_detection.p2_height_12 = p2_height_21



def set_measurements(dataset, detections):
	# Only need to go through left detections since Detection class contains stereo detection
	for i, detection in enumerate(detections[0]):
		if(detection.is_stereo):
			# Width
			width_3D = -1
			p1_width_bary_0 = None
			p1_width_bary_1 = None
			p2_width_bary_0 = None
			p2_width_bary_1 = None
			if(all(p != (-1,-1) for p in [detection.p1_width, detection.stereo_detection.p1_width_12, detection.p1_width_12, detection.stereo_detection.p1_width])):
				p1_width_bary_0 = (np.array(detection.p1_width) + np.array(detection.stereo_detection.p1_width_12))/2
				p1_width_bary_1 = (np.array(detection.p1_width_12) + np.array(detection.stereo_detection.p1_width))/2
				detection.p1_width_bary = p1_width_bary_0
				detection.stereo_detection.p1_width_bary = p1_width_bary_1
			if(all(p != (-1,-1) for p in [detection.p2_width, detection.stereo_detection.p2_width_12, detection.p2_width_12, detection.stereo_detection.p2_width])):
				p2_width_bary_0 = (np.array(detection.p2_width) + np.array(detection.stereo_detection.p2_width_12))/2
				p2_width_bary_1 = (np.array(detection.p2_width_12) + np.array(detection.stereo_detection.p2_width))/2
				#print(p2_width_bary_0, p2_width_bary_1)
				detection.p2_width_bary = p2_width_bary_0
				detection.stereo_detection.p2_width_bary = p2_width_bary_1
			if(all(p is not None for p in [p1_width_bary_0, p1_width_bary_1, p2_width_bary_0, p2_width_bary_1])):
				p1_width_3D = triangulate(dataset, p1_width_bary_0, p1_width_bary_1)
				p2_width_3D = triangulate(dataset, p2_width_bary_0, p2_width_bary_1)
				width_3D = np.linalg.norm(p1_width_3D-p2_width_3D)

			# Height
			height_3D = -1
			p1_height_bary_0 = None
			p1_height_bary_1 = None
			p2_height_bary_0 = None
			p2_height_bary_1 = None
			if(all(p != (-1,-1) for p in [detection.p1_height, detection.stereo_detection.p1_height_12, detection.p1_height_12, detection.stereo_detection.p1_height])):
				p1_height_bary_0 = (np.array(detection.p1_height) + np.array(detection.stereo_detection.p1_height_12))/2
				p1_height_bary_1 = (np.array(detection.p1_height_12) + np.array(detection.stereo_detection.p1_height))/2
				detection.p1_height_bary = p1_height_bary_0
				detection.stereo_detection.p1_height_bary = p1_height_bary_1
			if(all(p != (-1,-1) for p in [detection.p2_height, detection.stereo_detection.p2_height_12, detection.p2_height_12, detection.stereo_detection.p2_height])):
				p2_height_bary_0 = (np.array(detection.p2_height) + np.array(detection.stereo_detection.p2_height_12))/2
				p2_height_bary_1 = (np.array(detection.p2_height_12) + np.array(detection.stereo_detection.p2_height))/2
				detection.p2_height_bary = p2_height_bary_0
				detection.stereo_detection.p2_height_bary = p2_height_bary_1
			if(all(p is not None for p in [p1_height_bary_0, p1_height_bary_1, p2_height_bary_0, p2_height_bary_1])):
				p1_height_3D = triangulate(dataset, p1_height_bary_0, p1_height_bary_1)
				p2_height_3D = triangulate(dataset, p2_height_bary_0, p2_height_bary_1)
				height_3D = np.linalg.norm(p1_height_3D-p2_height_3D)

			# Center
			center_3D = (0, 0, 0)
			if(width_3D != -1) and (height_3D != -1):
				width_bary = (p1_width_3D + p2_width_3D)/2
				height_bary = (p1_height_3D + p2_height_3D)/2
				center_3D = (width_bary + height_bary)/2
				center_3D = center_3D.tolist()

				width_2D_bary_0 = (p1_width_bary_0 + p2_width_bary_0)/2
				height_2D_bary_0 = (p1_height_bary_0 + p2_height_bary_0)/2
				center_2D_0 = (width_2D_bary_0 + height_2D_bary_0)/2
				detection.center_2D = center_2D_0

				width_2D_bary_1 = (p1_width_bary_1 + p2_width_bary_1)/2
				height_2D_bary_1 = (p1_height_bary_1 + p2_height_bary_1)/2
				center_2D_1 = (width_2D_bary_1 + height_2D_bary_1)/2
				detection.stereo_detection.center_2D = center_2D_1

			elif(width_3D != -1):
				width_bary = (p1_width_3D + p2_width_3D)/2
				center_3D = width_bary
				center_3D = center_3D.tolist()

				center_2D_0 = (p1_width_bary_0 + p2_width_bary_0)/2
				detection.center_2D = center_2D_0
				center_2D_1 = (p1_width_bary_1 + p2_width_bary_1)/2
				detection.center_2D = center_2D_1


			elif(height_3D != -1):
				height_bary = (p1_height_3D + p2_height_3D)/2
				center_3D = height_bary
				center_3D = center_3D.tolist()

				center_2D_0 = (p1_height_bary_0 + p2_height_bary_0)/2
				detection.center_2D = center_2D_0
				center_2D_1 = (p1_height_bary_1 + p2_height_bary_1)/2
				detection.center_2D = center_2D_1

			measurement = Measurement(center=center_3D, height=height_3D, width=width_3D)
			detection.measurement = measurement
			detection.stereo_detection.measurement = measurement

