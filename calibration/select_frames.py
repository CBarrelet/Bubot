import argparse
from os import path
from os import mkdir
from shutil import rmtree
import cv2
import numpy as np

def action_key(key, frame_counter, frame_jump, n_frames, save, selection):
	escape = 27 # Escape selection
	q = 113 	# frame_jump *= 2
	a = 97  	# frame_jump /= 2
	d = 100 	# frame_count += frame_jump
	e = 101 	# frame_count -= frame_jump
	space = 32  # Save frame
	if(key == escape):
		selection = False
		print("Exit selection")
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
		frame_counter += frame_jump
		if(frame_counter > n_frames-30):
			# cv2.CAP_PROP_FRAME_COUNT doesn't return the actual last frame
			# -30 seems to work for every videos, only god knows exactly why though
			frame_counter = n_frames-30
	elif(key == q):	
		frame_counter -= frame_jump
		if(frame_counter < 0):
			frame_counter = 0
	elif(key == space):
		save = True
	return frame_counter, frame_jump, save, selection

def print_option():
	print("Key bindings :")
	print("'q' - Previous frame")
	print("'d' - Next frame")
	print("'a' - Divise step by 2")
	print("'d' - Multiply step by 2")
	print("'space' - Save current frames")
	print("'escape' - Exit selection")

def selection():
	left_video_path, right_video_path, output_path = opt.left, opt.right, opt.output

	frame_counter = 0
	frame_jump = 1
	saved_counter = 0
	selection = True

	if(path.exists(output_path)):
		rmtree(output_path)
	mkdir(output_path)

	left_video = cv2.VideoCapture(left_video_path)
	right_video = cv2.VideoCapture(right_video_path)

	left_n_frames = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
	right_n_frames = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))
	n_frames = min(left_n_frames, right_n_frames)

	print_option()

	while selection:
		save = False

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
		frame_counter, frame_jump, save, selection = action_key(key, frame_counter, frame_jump, n_frames, save, selection)

		if(save):
			left_path = output_path + "left_" + str(saved_counter) + ".png"
			right_path = output_path + "right_" + str(saved_counter) + ".png"
			saved_counter += 1
			cv2.imwrite(left_path, left_frame)
			cv2.imwrite(right_path, right_frame)
			print("Frame number %d saved"%frame_counter)
			print("%d frames has been saved"%saved_counter)

		left_video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
		right_video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

	left_video.release()
	right_video.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--left', type=str, help='Left video path')
	parser.add_argument('--right', type=str, help='Right video path')
	parser.add_argument('--output', type=str, help='Output directory path')
	opt = parser.parse_args()

	selection()