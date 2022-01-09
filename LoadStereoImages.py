import os
import glob
from pathlib import Path

from detection.utils.datasets import letterbox

from main_utils import read_json, check_json_files, read_pickle

import numpy as np
import cv2

class LoadStereoImages:  # for inference

    def __init__(self, opt, path, img_size=640, stride=32):
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        self.vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        p = str(Path(path).absolute())  # os-agnostic absolute path

        if '*' in p:
            json_files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            check_json_files(p)
            json_files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            json_files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        self.concat_out = opt.concat_output

        images_json_files = [x for x in json_files if read_json(x)['left_path'].split('.')[-1].lower() in self.img_formats]
        imagesL = [read_json(x)['left_path'] for x in json_files if read_json(x)['left_path'].split('.')[-1].lower() in self.img_formats]
        imagesR = [read_json(x)['right_path'] for x in json_files if read_json(x)['right_path'].split('.')[-1].lower() in self.img_formats]
        images_calibration_files = [read_json(x)['calibration_path'] for x in json_files if read_json(x)['left_path'].split('.')[-1].lower() in self.img_formats]

        videos_json_files = [x for x in json_files if read_json(x)['left_path'].split('.')[-1].lower() in self.vid_formats]
        videosL = [read_json(x)['left_path'] for x in json_files if read_json(x)['left_path'].split('.')[-1].lower() in self.vid_formats]
        videosR = [read_json(x)['right_path'] for x in json_files if read_json(x)['right_path'].split('.')[-1].lower() in self.vid_formats]
        videos_calibration_files = [read_json(x)['calibration_path'] for x in json_files if read_json(x)['left_path'].split('.')[-1].lower() in self.vid_formats]

        niL, nvL = len(imagesL), len(videosL)
        niR, nvR = len(imagesR), len(videosR)

        self.img_size = img_size
        self.stride = stride
        self.images_json_files = images_json_files
        self.videos_json_files = videos_json_files
        self.filesL = imagesL + videosL
        self.filesR = imagesR + videosR
        self.calibration_files = images_calibration_files + videos_calibration_files

        self.nf = niL + nvL  # number of file pairs

        self.video_flag = [False] * niL + [True] * nvL
        self.mode = 'image'

        if(any(videosL) and any(videosR)):
            self.new_video_pair(videosL[0], videosR[0])  # new video pair
            self.load_calibration(videos_calibration_files[0])
        else:
            self.capL = None
            self.capR = None

        # Concatened or left/right video path/writer
        self.vid_path, self.vid_writer = None, None
        self.vid_pathL, self.vid_writerL = None, None
        self.vid_pathR, self.vid_writerR = None, None

        assert self.nf > 0, f'No json file found in {p}.'


    def __iter__(self):
        self.count = 0
        return self


    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        pathL = self.filesL[self.count]
        pathR = self.filesR[self.count]
        calibration_path = self.calibration_files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_valL, img0L = self.capL.read()
            ret_valR, img0R = self.capR.read()

            if not(ret_valL) or not(ret_valR):
                self.count += 1
                self.capL.release()
                self.capR.release()
                if self.count == self.nf:  # last video pair
                    raise StopIteration
                else:
                    pathL = self.filesL[self.count]
                    pathR = self.filesR[self.count]
                    calibration_path = self.calibration_files[self.count]
                    self.new_video_pair(pathL, pathR)
                    self.load_calibration(calibration_path)

                    ret_valL, img0L = self.capL.read()
                    ret_valR, img0R = self.capR.read()

            self.frame += 1
            print(f'Video pair {self.count + 1}/{self.nf} \nLeft:  ({self.frame}/{self.nframesL}) {pathL} \nRight: ({self.frame}/{self.nframesR}) {pathR}\n', end='')

        else:
            # Read image
            img0L = cv2.imread(pathL)
            img0R = cv2.imread(pathR)  # BGR
            self.load_calibration(calibration_path)
            assert img0L is not None, f"\nImage '{pathL}' not found"
            assert img0R is not None, f"\nImage '{pathR}' not found"
            print(f'Image pair {self.count}/{self.nf} \nLeft:  {pathL} \nRight: {pathR}\n', end='')
            self.count += 1
        #

        if(self.mode == 'image'):
            json_path = self.images_json_files[self.count-1]
        else:
            json_path = self.videos_json_files[self.count-1]

        assert img0L.shape == img0R.shape, f"\nData don't share same shape.\n" \
                            f"Json file: {json_path}.\n" \
                            f"Left {self.mode}: {pathL}\n" \
                            f"Right {self.mode}: {pathR}\n" \
                            f"Left {self.mode} shape: {img0L.shape}\n" \
                            f"Right {self.mode} shape: {img0R.shape}\n" \


        # Rectification
        cv2.remap(img0L, self.stereo_params["mapx1"], self.stereo_params["mapy1"], cv2.INTER_LINEAR)
        cv2.remap(img0R, self.stereo_params["mapx2"], self.stereo_params["mapy2"], cv2.INTER_LINEAR)
        # Padded resize
        imgL = letterbox(img0L, self.img_size, stride=self.stride)[0]
        imgR = letterbox(img0R, self.img_size, stride=self.stride)[0]
        # Convert BGR to RGB, to 3x416x416
        imgL = imgL[:, :, ::-1].transpose(2, 0, 1)
        imgR = imgR[:, :, ::-1].transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL)
        imgR = np.ascontiguousarray(imgR)
        imgs = [imgL, imgR]
        img0s = [img0L, img0R]

        if(self.concat_out):
            paths = [Path(json_path).stem+ '_json' + Path(pathL).suffix]
        else:
            paths = [str(Path(json_path).stem) + '_json_' + Path(pathL).name, str(Path(json_path).stem) + '_json_' + Path(pathR).name]

        self.caps = [self.capL, self.capR]

        self.img0s = img0s
        self.paths = paths

        return imgs, img0s


    def new_video_pair(self, pathL, pathR):
        self.frame = 0
        self.capL = cv2.VideoCapture(pathL)
        self.capR = cv2.VideoCapture(pathR)
        self.nframesL = int(self.capL.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframesR = int(self.capR.get(cv2.CAP_PROP_FRAME_COUNT))


    def load_calibration(self, path):
        stereo_params = read_pickle(path)
        assert isinstance(stereo_params, dict), f"\nERROR: Loaded calibration file '{path}' is not a a dict"
        stereo_params.keys()
        assert 'mapx1' in stereo_params, f"\nERROR: Missing key 'mapx1' in calibration file '{path}'"
        assert 'mapy1' in stereo_params, f"\nERROR: Missing key 'mapy1' in calibration file '{path}'"
        assert 'mapx2' in stereo_params, f"\nERROR: Missing key 'mapx2' in calibration file '{path}'"
        assert 'mapy2' in stereo_params, f"\nERROR: Missing key 'mapy2' in calibration file '{path}'"
        assert 'P1' in stereo_params, f"\nERROR: Missing key 'P1' in calibration file '{path}'"
        assert 'P2' in stereo_params, f"\nERROR: Missing key 'P2' in calibration file '{path}'"
        self.stereo_params = stereo_params
        # TODO: Change calibration script
        self.stereo_params['square_size'] = 5

        
    def __len__(self):
        return self.nf  # number of file pairs


    def save_image(self, save_dir):
         # Save results (image with detections)
        if(self.concat_out):
            img0 = np.hstack((self.img0s[0], self.img0s[1]))
            save_path = str(save_dir/f'concat_out_{Path(self.paths[0]).name}')
        else:
            save_pathL = str(save_dir/Path(self.paths[0]).name)
            save_pathR = str(save_dir/Path(self.paths[1]).name)

        if self.mode == 'image':
            if(self.concat_out):
                cv2.imwrite(save_path, img0)
            else:
                cv2.imwrite(save_pathL, self.img0s[0])
                cv2.imwrite(save_pathR, self.img0s[1])

        else:  # 'video'
            if(self.concat_out):
                self.vid_path, self.vid_writer = self._write_vid(save_path, self.vid_path, self.vid_writer, self.caps[0], img0)
            else:
                self.vid_pathL, self.vid_writerL = self._write_vid(save_pathL, self.vid_pathL, self.vid_writerL, self.caps[0], self.img0s[0])
                self.vid_pathR, self.vid_writerR = self._write_vid(save_pathR, self.vid_pathR, self.vid_writerR, self.caps[1], self.img0s[1])

    def _write_vid(self, save_path, vid_path, vid_writer, vid_cap, img):
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h, w = img.shape[:2]
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(img)
        return vid_path, vid_writer

