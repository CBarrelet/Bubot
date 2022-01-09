import argparse
import cv2
import numpy as np
import glob
import pickle


def init3Dpoints(board_size):
	points_3d = np.zeros((board_size[0]*board_size[1],3), np.float32)
	points_3d[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
	return points_3d

def findCorners(img, board_size):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)
	ret, corners = cv2.findChessboardCorners(gray_img, board_size, flags)
	if(ret):
		if(corners[0][0][1] < corners[board_size[0]-1][0][1]):
			corners = corners[::-1]
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
		sub_corners = cv2.cornerSubPix(gray_img, np.array(corners), (11, 11), (-1, -1), criteria)
		corners = sub_corners
	return ret, corners

def showCorners(left_image, right_image, left_corners, right_corners, board_size):
	left_image = cv2.drawChessboardCorners(left_image, board_size, left_corners, True)
	right_image = cv2.drawChessboardCorners(right_image, board_size, right_corners, True)
	left_resized_image = cv2.resize(left_image, (640, 360), cv2.INTER_CUBIC);
	right_resized_image = cv2.resize(right_image, (640, 360), cv2.INTER_CUBIC);
	twin_image = np.hstack((left_resized_image, right_resized_image))
	cv2.imshow("Left - Right corners", twin_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def reprojectionErrors(left_image_points, right_image_points, object_points, mtxL, mtxR, distL, distR, rvecsL, rvecsR, tvecsL, tvecsR):
	tot_errorL = 0
	tot_errorR = 0
	errorL = 0
	errorR = 0
	for i in range(len(object_points)):
	    project_left_points, _ = cv2.projectPoints(object_points[i], rvecsL[i], tvecsL[i], mtxL, distL)
	    project_right_points, _ = cv2.projectPoints(object_points[i], rvecsR[i], tvecsR[i], mtxR, distR)
	    errorL = cv2.norm(left_image_points[i], project_left_points, cv2.NORM_L2)/len(project_left_points)
	    errorR = cv2.norm(right_image_points[i], project_right_points, cv2.NORM_L2)/len(project_right_points)
	    tot_errorL += errorL
	    tot_errorR += errorR
	print('Left mean error : ', tot_errorL/len(object_points))
	print('Right mean error : ', tot_errorR/len(object_points))

def stereoCalibration():

	calibration_output_path = "calibration_files/"

	left_images_paths = glob.glob("image_3/left_*")
	right_images_paths = glob.glob("image_3/right_*")

	#left_images_paths = glob.glob("image_4/Left*")
	#right_images_paths = glob.glob("image_4/Right*")

	#left_images_paths = glob.glob("images_8/left_*")
	#right_images_paths = glob.glob("images_8/right_*")

	left_images_paths = glob.glob("small_calibration_images/left_*")
	right_images_paths = glob.glob("small_calibration_images/right_*")

	left_images_paths = glob.glob("large_calibration_images/left_*")
	right_images_paths = glob.glob("large_calibration_images/right_*")

	# Inverted because videos were inverted...
	#right_images_paths = glob.glob("calibration_new_method/left_*")
	#left_images_paths = glob.glob("calibration_new_method/right_*")

	# Not inverted because videos were inverted...
	#left_images_paths = glob.glob("calibration_new_method_2/left_*")
	#right_images_paths = glob.glob("calibration_new_method_2/right_*")

	left_images_paths = glob.glob("calibration_new_method_3/left_*")
	right_images_paths = glob.glob("calibration_new_method_3/right_*")

	#left_images_paths = glob.glob("water_image/left_*")
	#right_images_paths = glob.glob("water_image/right_*")

	#left_images_paths = glob.glob("original_frames/l/*")
	#right_images_paths = glob.glob("original_frames/r/*")

	left_images_paths.sort()
	right_images_paths.sort()
	

	for l_path, r_path in zip(left_images_paths, right_images_paths):
		print(l_path, r_path)

	if(len(left_images_paths) != len(right_images_paths)):
		print("Error : Number of images not equal")

	display = False

	board_width = 41
	board_height = 21

	#board_width = 10
	#board_height = 7

	board_width = 15
	board_height = 10

	board_width = 7
	board_height = 4
	board_size = (board_height, board_width)

	object_points = []
	points_3d = init3Dpoints(board_size)

	left_image_points = []
	right_image_points = []

	for i in range(len(left_images_paths)):

		print("Frames %d"%i)

		left_image = cv2.imread(left_images_paths[i])
		right_image = cv2.imread(right_images_paths[i])

		#left_image = cv2.resize(left_image, (590, 960), cv2.INTER_CUBIC);
		#right_image = cv2.resize(right_image, (590, 960), cv2.INTER_CUBIC);


		left_ret, left_corners = findCorners(left_image, board_size)
		right_ret, right_corners = findCorners(right_image, board_size)

		if(left_ret and right_ret):
			print("May be ok :", left_images_paths[i])
			if(display):
				showCorners(left_image, right_image, left_corners, right_corners, board_size)
			left_image_points.append(left_corners)
			right_image_points.append(right_corners)
			object_points.append(points_3d)


	print("%d chessboards found"%len(object_points))

	width = left_image.shape[1]
	height = left_image.shape[0]
	imgSize = (width, height)
	print(imgSize)

	

	# Calibrate mono cameras
	print('Calibrate left camera...')
	ret, M1, D1, r1, t1 = cv2.calibrateCamera(object_points, left_image_points, imgSize, None, None, None)
	print(ret)
	print('Calibrate right camera...')
	ret, M2, D2, r2, t2 = cv2.calibrateCamera(object_points, right_image_points, imgSize, None, None, None)
	print(ret)

	# Calculate reprojection errors
	print('Calculating reprojection errors...')
	reprojectionErrors(left_image_points, right_image_points, object_points, M1, M2, D1, D2, r1, r2, t1, t2)

	Left_cam_params = {'M1':M1, 'D1':D1, 'r1':r1, 't1':t1}
	Right_cam_params = {'M2':M2, 'D2':D2, 'r2':r2, 't2':t2}

	print('Saving calibration parameters...')
	pickle.dump(Left_cam_params, open(calibration_output_path+'new_large_left_cam_params_3.p', 'wb'))
	pickle.dump(Right_cam_params, open(calibration_output_path+'new_large_right_cam_params_3.p', 'wb'))

	"""
	calibration_output_path = "calibration_files/"
	Stereo_params = pickle.load(open(calibration_output_path+'Julian_stereo_params.p', 'rb'))
	Stereo_params.keys()

	# Restore calibration parameters from loaded dictionary.

	M1 = Stereo_params['M1']
	D1 = Stereo_params['D1']
	M2 = Stereo_params['M2']
	D2 = Stereo_params['D2']
	"""
	print('Cacluating the rectify parameters for stereo cameras...')

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

	stereo_flags = 0
	#stereo_flags |= cv2.CALIB_FIX_INTRINSIC
	stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
	stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
	#stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
	stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
	#stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
	#stereo_flags |= cv2.CALIB_RATIONAL_MODEL
	#stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
	#stereo_flags |= cv2.CALIB_FIX_K3
	#stereo_flags |= cv2.CALIB_FIX_K4
	#stereo_flags |= cv2.CALIB_FIX_K5

	ret, rectified_M1, rectified_D1, rectified_M2, rectified_D2, R, T, E, F = cv2.stereoCalibrate(object_points, left_image_points, right_image_points, 
																										M1, D1, M2, D2, imgSize, 
																										criteria, flags=stereo_flags)
	print("Stereo RMS : ", ret)
	print(" M1 : \n", M1)
	print(" M2 :\n", M2)
	print("rectified M1 : \n", rectified_M1)
	print("rectified M2 :\n", rectified_M2)

	# Stereo rectify
	R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(rectified_M1, rectified_D1, rectified_M2, rectified_D2, imgSize, R, T, flags=0)

	print("T =\n", T)

	print('validPixROI1 = ')
	print(validPixROI1)
	print('validPixROI2 = ')
	print(validPixROI2)

	mapx1, mapy1 = cv2.initUndistortRectifyMap(rectified_M1, rectified_D1, R1, P1, imgSize, cv2.CV_32FC1)
	mapx2, mapy2 = cv2.initUndistortRectifyMap(rectified_M2, rectified_D2, R2, P2, imgSize, cv2.CV_32FC1)

	Stereo_params = {'M1':M1, 'D1':D1, 'M2':M2, 'D2':D2, 'R1':R1, 'P1':P1, 'R2':R2, 'P2':P2, 
						'validPixROI1':validPixROI1, 'validPixROI2':validPixROI2, 
						'mapx1':mapx1, 'mapy1':mapy1, 'mapx2':mapx2, 'mapy2':mapy2, 
						'R':R, 'T':T, 'E':E, 'F':F, 'Q':Q}

	print('Saving stereo calibration parameters...')
	pickle.dump(Stereo_params, open(calibration_output_path+'new_large_stereo_params_3.p', 'wb'))

	calibration_output_path = "calibration_files/"
	Stereo_params = pickle.load(open(calibration_output_path+'new_large_stereo_params_3.p', 'rb'))
	Stereo_params.keys()

	# Restore calibration parameters from loaded dictionary.
	mapx1 = Stereo_params['mapx1']
	mapy1 = Stereo_params['mapy1']
	mapx2 = Stereo_params['mapx2']
	mapy2 = Stereo_params['mapy2']
	validPixROI1 = Stereo_params['validPixROI1']
	validPixROI2 = Stereo_params['validPixROI2']


	for i in range(len(left_images_paths)):
		left_image = cv2.imread(left_images_paths[i])
		right_image = cv2.imread(right_images_paths[i])

		dstL = cv2.remap(left_image, mapx1, mapy1, cv2.INTER_LINEAR)
		dstR = cv2.remap(right_image, mapx2, mapy2, cv2.INTER_LINEAR)

		dstL = cv2.rectangle(dstL, (validPixROI1[0], validPixROI1[1]), (validPixROI1[2], validPixROI1[3]), (0,0,255), 2) 
		dstR = cv2.rectangle(dstR, (validPixROI2[0], validPixROI2[1]), (validPixROI2[2], validPixROI2[3]), (0,0,255), 2) 

		left_resized_image = cv2.resize(dstL, (960, 590), cv2.INTER_CUBIC);
		right_resized_image = cv2.resize(dstR, (960, 590), cv2.INTER_CUBIC);
		twin_image = np.hstack((left_resized_image, right_resized_image))

		for i in range(0, twin_image.shape[0], 16):
			twin_image = cv2.line(twin_image, (0, i), (twin_image.shape[1], i), (0,255,0), 1)

		cv2.imshow("twin", twin_image)
		k = cv2.waitKey()
		if k==27:    # Esc key to stop
			break
		cv2.destroyAllWindows()



if __name__ == '__main__':
	stereoCalibration()





