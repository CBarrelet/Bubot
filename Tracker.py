from tracking.deep_sort.tracker import Tracker as Deep_Tracker
from tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric

from Track import Track
from tracking.deep_sort.track import TrackState


class Tracker(Deep_Tracker):

	def __init__(self, metric='cosine', matching_threshold=0.4, max_iou_distance=0.7, max_age=30, n_init=3):
		
		self.metric = NearestNeighborDistanceMetric(metric, matching_threshold=matching_threshold)
		super().__init__(self.metric, max_iou_distance, max_age, n_init)

	def _initiate_track(self, detection):
		mean, covariance = self.kf.initiate(detection.to_xyah())
		self.tracks.append(Track(detection,
			mean, covariance, self._next_id, self.n_init, self.max_age,
			detection.feature))
		self._next_id += 1

	def get_active(self):
		return [track for track in self.tracks if(track.state == TrackState.Confirmed)]

"""
def stereo_track_pariring(trackerL, trackerR):
	for trackL in trackerL.get_active():
		detectionL = trackL.get_last_detection()
		if(detectionL.is_stereo):
			for trackR in trackerR.get_active():
				detectionR = trackR.get_last_detection()
				if(detectionR.is_stereo):
					if((detectionL.measurement.center == detectionR.measurement.center).all()):
						if(not(trackL.is_stereo) and not(trackR.is_stereo)):
							trackL.is_stereo = True
							trackR.is_stereo = True
							trackL.stereo_id = trackL.track_id
							trackR.stereo_id = trackL.track_id
							trackL.color = detectionL.stereo_color
							trackR.color = trackL.color
							print(trackL.color,trackR.color)
							break
"""
def stereo_track_pariring(trackerL, trackerR):
	for trackL in trackerL.get_active():
		if(trackL.is_stereo):
			posL = trackL.tracked_3D_pos[-1]

			for trackR in trackerR.get_active():
				if(trackR.is_stereo):
					posR = trackR.tracked_3D_pos[-1]

					if((posL == posR).all()):
						stereo_id = trackL.track_id
						trackL.stereo_id = stereo_id
						trackR.stereo_id = stereo_id

						"""
						if(not(trackL.is_colored) and not(trackR.is_colored)):
							trackL.stereo_id = trackL.track_id
							trackR.stereo_id = trackL.track_id


							#trackL.color = trackL.get_last_detection().stereo_color
							trackR.color = trackL.color
							trackL.is_colored = True
							trackR.is_colored = True
						"""
							