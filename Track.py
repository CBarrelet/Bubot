import numpy as np
from tracking.deep_sort.track import Track as Deep_Track
from tracking.deep_sort.track import TrackState

from reconstruction.reconstruction import Measurement

class Track(Deep_Track):

	def __init__(self, detection, mean, covariance, track_id, n_init, max_age,
				feature=None):

		super().__init__(mean, covariance, track_id, n_init, max_age, feature=feature)

		self.color = detection.stereo_color if detection.is_stereo else detection.color
		self.is_stereo = False

		self.stereo_id = self.track_id

		self.detections = []
		if detection is not None:
			self.detections.append(detection)

		# 3D animation
		self.is_colored = False

		self.first_frame = -1
		self.tracked_2D_pos = []
		self.tracked_3D_pos = []
		self.interpolated_3D_pos = []
		self.tracked_height = []
		self.tracked_width = []
		self.mean_height = 0
		self.mean_width = 0
		self.tracked_top5 = []
		self.name = ''

		self.test = True


	def update(self, kf, detection):
		"""Perform Kalman filter measurement update step and update the feature
		cache.

		Parameters
		----------
		kf : kalman_filter.KalmanFilter
		    The Kalman filter.
		detection : Detection
		    The associated detection.

		"""
		self.mean, self.covariance = kf.update(
		    self.mean, self.covariance, detection.to_xyah())
		self.features.append(detection.feature)

		self.detections.append(detection)

		self.hits += 1
		self.time_since_update = 0
		if self.state == TrackState.Tentative and self.hits >= self._n_init:
			self.state = TrackState.Confirmed
		
		if(self.state == TrackState.Confirmed):

			if(self.test):
				self.first_frame = detection.frame_n
				self.test = False
				self.color = detection.classification.color

			if(detection.is_stereo):
				if(not(len(self.tracked_3D_pos))):
					self.first_frame = detection.frame_n

			if detection.is_stereo and not(self.is_stereo):
				self.color = detection.stereo_color 
				self.is_stereo = True


			if(self.first_frame != -1):
				self.tracked_top5.append(detection.classification.top5)
				self.tracked_2D_pos.append(np.array(self.to_tlbr(), dtype=int))
				if(detection.is_stereo):
					self.is_stereo = True
					center = np.array([detection.measurement.center[0],detection.measurement.center[1],detection.measurement.center[2]], dtype=float)
					self.tracked_3D_pos.append(center)
					self.tracked_height = detection.measurement.height
					self.tracked_width = detection.measurement.width
				else:
					center = np.array([0,0,0], dtype=float)
					self.tracked_3D_pos.append(center)


	def mark_missed(self):
		"""Mark this track as missed (no association at the current time step).
		"""
		if(self.state == TrackState.Confirmed):
			if(self.first_frame != -1):
				if(self.is_stereo):
					center = np.array([0,0,0], dtype=float)
					self.tracked_3D_pos.append(center)
					self.tracked_2D_pos.append(np.array(self.to_tlbr(), dtype=int))

		if self.state == TrackState.Tentative:
			self.state = TrackState.Deleted
		elif self.time_since_update > self._max_age:
			self.state = TrackState.Deleted


	def get_last_detection(self):
		detection = self.detections[-1]
		detection.xyxy = np.array(self.to_tlbr(), dtype=int)
		detection.stereo_color = self.color
		detection.id = self.stereo_id
		return detection


	def update_animation_settings(self):
		# Get median mesurements
		self.mean_width = np.median(self.tracked_width)
		self.mean_height = np.median(self.tracked_height)
		self.update_name()
		self.update_pos()


	def update_name(self):
		# Get best label from tracked top5
		top5_dict = {}
		for top5 in self.tracked_top5:
			for i, label in enumerate(top5[0]):
				if(label in top5_dict):
					top5_dict[label] += top5[1][i]
				else:
					top5_dict[label] = top5[1][i]
		if(top5_dict):
			self.name = max(top5_dict, key=top5_dict.get)


	def update_pos(self):
		# Start pos with non zero
		new_tracked_3D_pos = []
		first_nonzero = False
		if(np.count_nonzero(self.tracked_3D_pos[0])):
			pass
		else:
			for pos in self.tracked_3D_pos:
				if(np.count_nonzero(pos)) and not(first_nonzero):
					new_tracked_3D_pos.append(pos)
					first_nonzero = True
				if(first_nonzero):
					new_tracked_3D_pos.append(pos)
			self.tracked_3D_pos = new_tracked_3D_pos

		self.interpolated_3D_pos = []

		# Interpolate 3D pos from tracked pos if bbx has no measurement
		for i, pos in enumerate(self.tracked_3D_pos):
			if(np.count_nonzero(pos)):
				self.interpolated_3D_pos.append(pos)
			else:				
				pos_0 = self.interpolated_3D_pos[i-1]
				interpolated = False
				for j, pos_n in enumerate(self.tracked_3D_pos[i:]):
					if(np.count_nonzero(pos_n)):
						new_pos = pos_0+(pos_n-pos_0)/(j+1)
						self.interpolated_3D_pos.append(new_pos)
						interpolated = True
					if(interpolated):
						break
				if(not(interpolated)):
					# No tracked measurement left 
					break






