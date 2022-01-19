import os
import glob
import shutil
import pickle
import json
import argparse1

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
pickle_formats = ['p', 'pkl', 'pickle']

def read_json(filepath):
	with open(filepath, 'r') as f:
		data = json.load(f)
	return data

def write_json(filepath, data):
	with open(filepath, 'w') as f:
		json.dump(data, f, indent='\t')

def read_pickle(filepath):
	with open(filepath, 'rb') as f:
		data = pickle.load(f)
	return data

def write_pickle(filepath, data):
	with open(filepath, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def create_output_dir(output_dir, remove=False):
	if(os.path.exists(output_dir)):
		if(remove):			
			shutil.rmtree(output_dir)
			print(f"INFO: Output directory '{os.path.abspath(output_dir)}' deleted.")
			os.mkdir(output_dir)
			print(f"INFO: Output directory '{os.path.abspath(output_dir)}' created.")	
		else:
			print(f"INFO: Output directory '{os.path.abspath(output_dir)}' already exists.")
	else:
		os.mkdir(output_dir)
		print(f"INFO: Output directory '{os.path.abspath(output_dir)}' created.")

def check_save_path(filepath):
	parent_dir_path = os.path.abspath(os.path.join(filepath, os.pardir))
	paths = glob.glob(parent_dir_path+'/*')
	for path in paths:
		if(os.path.basename(path) == os.path.basename(filepath)):
			print(f"WARNING: '{filepath}' already exists. File changed.")

def check_valid_options(opt):
	# Check names
	assert not('/' in opt.place), f"\nERROR: Place '{opt.place}' contains illegal caracter ('/')"
	assert not('/' in opt.date), f"\nERROR: Date '{opt.date}' contains illegal caracter ('/').\nDate format: dd_mm_yyyy"
	assert not('/' in opt.time), f"\nERROR: Time '{opt.time} contains illegal caracter ('/').\nDate format: hh_mm_ss"
	# Check images/videos left/right path
	assert os.path.isfile(opt.left_path), f"\nERROR: No left file found in '{os.path.abspath(opt.left_path)}'"
	assert os.path.isfile(opt.right_path), f"\nERROR: No right file found in '{os.path.abspath(opt.right_path)}'"
	left_filename, left_ext = os.path.splitext(os.path.basename(opt.left_path))
	right_filename, right_ext = os.path.splitext(os.path.basename(opt.right_path))
	assert (left_ext[1:].lower() in img_formats) or (left_ext[1:].lower() in vid_formats), f"\nERROR: {opt.left_path} is not a valid format.\nSupported formats are: \nImage: {img_formats}\nVideo: {vid_formats}"
	assert (right_ext[1:].lower() in img_formats) or (right_ext[1:].lower() in vid_formats), f"\nERROR: {opt.left_path} is not a valid format.\nSupported formats are: \nImage: {img_formats}\nVideo: {vid_formats}"
	assert opt.left_path != opt.right_path, f"\nERROR: Left and right paths are the same.\nLeft path: '{os.path.abspath(opt.left_path)}'\nRight path: '{os.path.abspath(opt.right_path)}'"
	if(left_filename == right_filename):
		print(f"WARNING: Both files seem to have the same name.\nLeft file name: '{os.path.basename(opt.left_path)}'\nRight file name: '{os.path.basename(opt.left_path)}'")
	assert left_ext.lower() == right_ext.lower(), f"\nERROR: Left file '{os.path.basename(opt.left_path)} and right file '{os.path.basename(opt.right_path)}' don't share same extension."
	# Check calibration file path
	assert os.path.isfile(opt.calibration_path), f"\nERROR: No calibration file found in '{os.path.abspath(opt.calibration_path)}'"
	filename, ext = os.path.splitext(os.path.basename(opt.calibration_path))
	assert ext[1:].lower() in pickle_formats, f"\nERROR: '{os.path.abspath(opt.calibration_path)}'' is not a valid format.\nSupported formats are: \nPickle: {pickle_formats}"
	if(not(opt.output_dir[-1] == '/')): opt.output_dir += '/'

def verify_json(dir_path):
	assert os.path.isdir(dir_path), f"\nERROR: '{os.path.abspath(opt.left_path)}' is not a directory."
	json_paths = glob.glob(dir_path+'/*.json')
	assert len(json_paths)>0, f"\nERROR: No json file found in '{os.path.abspath(dir_path)}'."
	data_list = []
	left_paths = []
	right_paths = []
	for path in json_paths:
		data = read_json(path)
		data_list.append((data, path))
		left_paths.append(data['left_path'])
		right_paths.append(data['right_path'])
	# Check occurence
	for i, (data, path) in enumerate(data_list):
		for data_, path_ in data_list[i+1:]:
			if(data['left_path'] == data_['left_path']) and (path != path_):
				print(f"WARNING: Conflict between '{path}' and '{path_}'. Same left path shared.")
		for data_, path_ in data_list[i+1:]:
			if(data['right_path'] == data_['right_path']) and (path != path_):
				print(f"WARNING: Conflict between '{path}' and '{path_}'. Same right path shared.")

def create_json(opt):
	create_output_dir(output_dir=opt.output_dir, remove=opt.rm)
	check_valid_options(opt)
	data = {}
	data['place'] = opt.place
	data['date'] = opt.date
	data['time'] = opt.time
	data['comment'] = opt.comment
	data['left_path'] = os.path.abspath(opt.left_path)
	data['right_path'] = os.path.abspath(opt.right_path)
	data['calibration_path'] = os.path.abspath(opt.calibration_path)
	filepath = f'{opt.output_dir}{opt.place}_{opt.date}_{opt.time}.json'
	check_save_path(filepath)
	write_json(filepath, data)
	print(f"INFO: Json file saved in '{filepath}'")
	verify_json(opt.output_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--place', type=str, default='Unknown', help='Shooting place: Place')
	parser.add_argument('--date', type=str, default='13_12_2000', help='Shooting date: dd_mm_yyyy')
	parser.add_argument('--time', type=str, default='12_00_00', help='Shooting date: hh_mm_ss')
	parser.add_argument('--left_path', type=str, default='filename_left.MP4', help='Left data path: /path/to/left_data.(jpg/mp4/...)')
	parser.add_argument('--right_path', type=str, default='filename_left.MP4', help='Right data path: /path/to/right_data.(jpg/mp4/...)')
	parser.add_argument('--calibration_path', type=str, default='calibration.p', help='Calibration data path: /path/to/right_data.p')
	parser.add_argument('--comment', type=str, default='No comment', help='Comment: Suny day, lot of fishes, no turbidity, etc.')
	parser.add_argument('--output_dir', type=str, default='./json_files', help='Output folder path')
	parser.add_argument('--rm', action='store_true', help='Remove output dir and create a new one')
	opt = parser.parse_args()

	create_json(opt)

