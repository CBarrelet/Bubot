import os
import shutil
import glob
import json
import pickle

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