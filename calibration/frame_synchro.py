import argparse
from os import path
from os import mkdir
from shutil import rmtree
from subprocess import call
from subprocess import run
from subprocess import PIPE
from subprocess import STDOUT
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import cv2
import numpy as np

def fast_extract_video_and_save(input_video_path, start, end, output_video_path):
	# Found at https://stackoverflow.com/questions/52257731/extract-part-of-a-video-using-ffmpeg-extract-subclip-black-frames
    # Makes a new video file playing video file ``filename`` between
    # the times ``t1`` and ``t2``.
    command = [get_setting("FFMPEG_BINARY"),"-y",
           "-ss", "%0.2f"%start,
           "-i", input_video_path,
           "-t", "%0.2f"%(end-start),
           "-vcodec", "copy", "-acodec", "copy", output_video_path]
    subprocess_call(command)

def accurate_extract_video_and_save(input_video_path, start, end, output_video_path):
	video = VideoFileClip(input_video_path).subclip(start, end)
	video.write_videofile(output_video_path)

def get_duration(video_path):
	# Found at https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
	result = run(["ffprobe", "-v", "error", "-show_entries",
								"format=duration", "-of",
								"default=noprint_wrappers=1:nokey=1", video_path],
		stdout=PIPE,
		stderr=STDOUT)
	return float(result.stdout)

def show_video_infos(video_path):
	print("Infos :", video_path)
	duration = get_duration(video_path)
	video = VideoFileClip(video_path)
	n_frames = video.reader.nframes
	print("Duration : %0.3fs"%duration)
	print("Frames :", n_frames)
	print("Fps : %0.2f"%video.fps)

def print_option():
	print("Key bindings :")
	print("'q' - Previous frame")
	print("'d' - Next frame")
	print("'a' - Divise step by 2")
	print("'d' - Multiply step by 2")
	print("'w' - Left camera selection")
	print("'x' - Both cameras selection")
	print("'c' - Right camera selection")
	print("'space' - Synchronize")
	print("'escape' - Exit selection")

def action_key(key, left_frame_counter, right_frame_counter, frame_jump, max_frames, cam_select, synchronize, selection):
	escape = 27 # Escape selection
	q = 113 	# frame_jump *= 2
	a = 97  	# frame_jump /= 2
	d = 100 	# frame_count += frame_jump
	e = 101 	# frame_count -= frame_jump
	w = 119		# only right frame
	x = 120		# both frames
	c = 99		# right frame
	space = 32  # Save frame

	if(key == escape):
		selection = False
		print("Exit selection")

	elif(key == w):
		print("Left camera selection")
		cam_select = [True, False, False]
	elif(key == x):
		print("Both cameras selection")
		cam_select = [False, True, False]
	elif(key == c):
		print("Right camera selection")
		cam_select = [False, False, True]

	elif(key == e):
		frame_jump *= 2
		print("Set frame jump : ", frame_jump)

	elif(key == a):
		frame_jump /= 2
		frame_jump = int(frame_jump)
		if(frame_jump<1):
			frame_jump = 1
		print("Set frame jump : ", frame_jump)

	elif(key == d):
		if(cam_select[0]):
			left_frame_counter += frame_jump
		if(cam_select[1]):
			right_frame_counter += frame_jump
			left_frame_counter += frame_jump
		if(cam_select[2]):
			right_frame_counter += frame_jump
		if(right_frame_counter > max_frames-30):
			# cv2.CAP_PROP_FRAME_COUNT doesn't return the actual last frame
			# -30 seems to work for every videos, only god knows exactly why though
			right_frame_counter = max_frames-30
		if(left_frame_counter > max_frames-30):
			left_frame_counter = max_frames-30

	elif(key == q):	
		if(cam_select[0]):
			left_frame_counter -= frame_jump
		if(cam_select[1]):
			right_frame_counter -= frame_jump
			left_frame_counter -= frame_jump
		if(cam_select[2]):
			right_frame_counter -= frame_jump
		if(right_frame_counter < 0):
			right_frame_counter = 0
		if(left_frame_counter < 0):
			left_frame_counter = 0

	elif(key == space):
		print("Left frame counter : %d"%left_frame_counter)
		print("Right frame counter : %d"%right_frame_counter)
		synchronize = True
	return left_frame_counter, right_frame_counter, frame_jump, cam_select, synchronize, selection

	def get_duration(video_path):
		# Found at https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
		result = run(["ffprobe", "-v", "error", "-show_entries",
									"format=duration", "-of",
									"default=noprint_wrappers=1:nokey=1", video_path],
			stdout=PIPE,
			stderr=STDOUT)
		return float(result.stdout)

def synchronization():
	video_left_path, video_right_path, time_cut, synchro = opt.left, opt.right, opt.cut, opt.synchro

	video_left_name = path.basename(video_left_path)
	video_left_name, left_extension = path.splitext(video_left_name)
	video_right_name = path.basename(video_right_path)
	video_right_name, right_extension = path.splitext(video_right_name)
	if(left_extension != right_extension):
		print("Error : Videos don't have the same extension")
		return 0
	else: extension = left_extension

	video_left_temp_path = "video/temp/temp_left" + extension
	video_left_synchro_temp_path = "video/temp/" + video_left_name + "_synchronized_temp" + extension
	video_left_synchro_path = "video/synchronized_" + video_left_name + extension
	video_right_temp_path = "video/temp/temp_right" + extension
	video_right_synchro_temp_path = "video/temp/" + video_right_name + "_synchronized_temp" + extension
	video_right_synchro_path = "video/synchronized_" + video_right_name + extension

	# Creating temporary direcctory
	if(path.exists("video/temp")):
		rmtree("video/temp")
	mkdir("video/temp")

	# Extracting the beginning of both videos
	fast_extract_video_and_save(video_left_path, 0, time_cut, video_left_temp_path)
	fast_extract_video_and_save(video_right_path, 0, time_cut, video_right_temp_path)

	left_video = cv2.VideoCapture(video_left_temp_path)
	right_video = cv2.VideoCapture(video_right_temp_path)

	left_n_frames = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
	right_n_frames = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))
	max_frames = min(left_n_frames, right_n_frames)

	fps = left_video.get(cv2.CAP_PROP_FPS)

	left_frame_counter = 0
	right_frame_counter = 0
	frame_jump = 1

	selection = True

	r_cam_select = False # Only left frame
	l_cam_select = False # Only right frame
	b_cam_select = True  # Both frames
	cam_select = (l_cam_select, b_cam_select, r_cam_select)

	print_option()
	while selection:
		synchronize = False

		ret_left, left_frame = left_video.read()
		ret_right, right_frame = right_video.read()

		if(ret_left == False) or (ret_right == False):
			print("Error : video path doesn't exist")
			return 0
		left_resized_frame = cv2.resize(left_frame, (640, 360), cv2.INTER_CUBIC);
		right_resized_frame = cv2.resize(right_frame, (640, 360), cv2.INTER_CUBIC);
		twin_frames = np.hstack((left_resized_frame, right_resized_frame))
		cv2.imshow("Left - Right frames", twin_frames)

		key = cv2.waitKey()
		left_frame_counter, right_frame_counter, frame_jump, cam_select, synchronize, selection = action_key(key, left_frame_counter, right_frame_counter, frame_jump, max_frames, cam_select, synchronize, selection)

		left_video.set(cv2.CAP_PROP_POS_FRAMES, left_frame_counter)
		right_video.set(cv2.CAP_PROP_POS_FRAMES, right_frame_counter)

		if(synchronize):
			cv2.destroyAllWindows()
			left_video = cv2.VideoCapture(video_left_path)
			right_video = cv2.VideoCapture(video_right_path)
			left_n_frames = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
			right_n_frames = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))
			max_frames = min(left_n_frames, right_n_frames)
			fast_extract_video_and_save(video_left_path, left_frame_counter/fps, max_frames/fps, video_left_synchro_path)
			fast_extract_video_and_save(video_right_path, right_frame_counter/fps, max_frames/fps, video_right_synchro_path)
			print("Videos synchronized and saved.")
			selection = False

	# Deleting temp directory
	rmtree("video/temp")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--left', type=str, help='Left video path')
	parser.add_argument('--right', type=str, help='Right video path')
	parser.add_argument('--cut', type=float, default=15 ,help='Time cut for synchro in seconds')
	parser.add_argument('--synchro', dest='synchro', action='store_true')
	parser.add_argument('--no-synchro', dest='synchro', action='store_false')
	parser.set_defaults(synchro=True)
	opt = parser.parse_args()

	synchronization()