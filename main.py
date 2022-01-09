import argparse
import time
from pathlib import Path

import cv2
from numpy import hstack
import numpy as np
import torch

from detection.utils.general import check_requirements, set_logging, increment_path
from detection.utils.torch_utils import select_device, time_synchronized

from main_utils import draw_results, draw_stereo_results, draw_dimensions, draw_least_dist, draw_all_dist, draw_segmentation, draw_length, draw_trajectories
from reconstruction.reconstruction import stereo_matching, set_measurements, set_pair_points

from LoadStereoImages import LoadStereoImages
from Detector import Detector
from Classifier import Classifier
from Segmentor import Segmentor
from Tracker import Tracker, stereo_track_pariring

from main_utils import save_pickle


def detect(opt):
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialization
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load detection model
    detector = Detector(path=opt.weights_detection, device=device, half=half, imgsz=opt.img_size, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)

    # Load classification model
    classifier = Classifier(path=opt.weights_classification, device=device, names_path=opt.classif_names)

    # Load segmentation model
    segmentor = Segmentor(path=opt.weights_segmentation, device=device)

    # Load dataset
    dataset = LoadStereoImages(opt=opt, path=opt.source, img_size=detector.imgsz, stride=detector.stride)

    # Initialize trackers
    trackerL = Tracker(metric='cosine', matching_threshold=0.4, max_iou_distance=0.7, max_age=30, n_init=3)
    trackerR = Tracker(metric='cosine', matching_threshold=0.4, max_iou_distance=0.7, max_age=30, n_init=3)

    t0 = time.time()

    for frame_n, (imgs, img0s) in enumerate(dataset):

        t1 = time_synchronized()
        # Fish detection
        detectionsL, log_detectionL = detector.detect(dataset=dataset, img_padded=imgs[0], img_shape=img0s[0].shape, frame_n=frame_n)
        detectionsR, log_detectionR = detector.detect(dataset=dataset, img_padded=imgs[1], img_shape=img0s[1].shape, frame_n=frame_n)
        detections = [detectionsL, detectionsR]

        t2 = time_synchronized()
        # Species identification
        classifier.classify(img0s, detections)

        t3 = time_synchronized()
        # Detection stero matching
        stereo_matching(detections)

        t35 = time_synchronized()
        for i, detection in enumerate(detectionsL):
            if(detection.is_stereo):

                tseg0 = time.time()
                # Segmentation
                segmentationL = segmentor.segment(img=img0s[0], detections=detection)
                segmentationR = segmentor.segment(img=img0s[1], detections=detection.stereo_detection)
                
                detection.segmentation = segmentationL
                detection.stereo_detection.segmentation = segmentationR

                # Get endpoints with PCA
                axis_1L, axis_2L, centerL = segmentor.get_endpoints(segmentationL)
                axis_1R, axis_2R, centerR = segmentor.get_endpoints(segmentationR)

                detection.set_endpoints(axis_1L, axis_2L, centerL)
                detection.stereo_detection.set_endpoints(axis_1R, axis_2R, centerR)

                set_pair_points(detection)


        t4 = time_synchronized()
        set_measurements(dataset, detections)


        t5 = time_synchronized()
        trackerL.predict()
        trackerR.predict()
        trackerL.update([detection for detection in detections[0]])
        trackerR.update([detection for detection in detections[1]])


        t6 = time_synchronized()
        if(dataset.mode == "video"):
            detectionsL = []
            detectionsR = []
            detections = []
            detectionsL = [track.get_last_detection() for track in trackerL.get_active()]
            detectionsR = [track.get_last_detection() for track in trackerR.get_active()]
            detections = [detectionsL, detectionsR]

        # Draw results
        if(not opt.nosave):
            img0s[0] = draw_trajectories(opt, img0s[0], trackerL)
            img0s[1] = draw_trajectories(opt, img0s[1], trackerR)

            if(opt.draw_stereo):
                draw_stereo_results(opt, img0s, detections)
            elif(opt.draw_normal):
                draw_results(opt, img0s[0], detections[0])
                draw_results(opt, img0s[1], detections[1])
            if(opt.draw_least):
                draw_least_dist(img0s, bboxes, paired_list, measurements, paired_colors)
            if(opt.draw_all_dist):
                draw_all_dist(img0s, bboxes, paired_list, measurements, paired_colors)

            img0s[0] = draw_segmentation(opt, img0s[0], detectionsL)
            img0s[1] = draw_segmentation(opt, img0s[1], detectionsR)
            img0s[0] = draw_length(opt, img0s[0], detectionsL)
            img0s[1] = draw_length(opt, img0s[1], detectionsR)


        t7 = time_synchronized()

        if(opt.show):
            img0 = hstack((img0s[0], img0s[1]))
            #cv2.imshow("Concat result", img0)
            cv2.imshow("Left result", img0s[0])
            #cv2.imshow("right result", img0s[1])
            cv2.waitKey()
            cv2.destroyAllWindows()

        if(not opt.nosave):
            dataset.save_image(save_dir)
        
        t8 = time_synchronized()

        print(f'Left:  {log_detectionL} - Right: {log_detectionR} - Det done: {t2 - t1:.3f}s - Classif done: {t3 - t2:.3f}s - Pair done: {t35 - t3:.3f}s - Seg done: {t4 - t35:.3f}s - 3D done: {t5 - t4:.3f}s - Tracking done: {t6 - t5:.3f} - Draw done: {t7 - t6:.3f}s - Save done: {t8 - t7:.3f}s')

    print(f'Done. ({time.time() - t0:.3f}s)')


    t = [track.update_animation_settings() for track in trackerL.tracks if track.is_stereo]
    print(len(t))

    save_pickle('tracker.p', trackerL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--classif_names', default='./classification/data/class_names.p', help='Path to the classifier labels')
    parser.add_argument('--concat_output', action='store_true', help='Concatenate outputs')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--draw_all_dist', action='store_true', help='Draw all distances between bboxes')
    parser.add_argument('--draw_dim', action='store_true', help='Draw bboxes dimensions')
    parser.add_argument('--draw_least', action='store_true', help='Draw least distances between bboxes')
    parser.add_argument('--draw_length', default=True, action='store_false', help='Draw length line')
    parser.add_argument('--draw_normal', action='store_true', help='Draw normal bbx')
    parser.add_argument('--draw_segmentation', default=True, action='store_false', help='Draw segmentation')
    parser.add_argument('--draw_stereo', action='store_true', help='Draw only paired bbx')
    parser.add_argument('--draw_trajectories', default=True, action='store_false', help='Draw trajectories')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--oppacity', type=float, default=1., help='Apply oppacity to drawings')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--weights_classification', default='./classification/weights/best_classification.pt', help='Path to the EfficientNet model')
    parser.add_argument('--weights_detection', nargs='+', type=str, default='./detection/weights/best_detection.pt', help='Path to the YoloV5 model')
    parser.add_argument('--weights_segmentation', default='./segmentation/weights/best_segmentation.pt', help='Path to the GoogleDeepLabV3 model')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        detect(opt=opt)
