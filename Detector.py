import numpy as np
import torch
from detection.models.experimental import attempt_load
from detection.utils.general import check_img_size, non_max_suppression, scale_coords
from Detection import Detection

class Detector(object):

    def __init__(self, path, device, half, imgsz, conf_thres, iou_thres):
        self.path = path
        self.device = device
        self.model = attempt_load(path, map_location=device) # load FP32 model
        if half: self.model.half()
        self.half = half
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = {name: [np.random.randint(0, 255) for _ in range(3)] for name in self.names}


    def detect(self, dataset, img_padded, img_shape, frame_n):
        img_padded = torch.from_numpy(img_padded).to(self.device)
        img_padded = img_padded.half() if self.half else img_padded.float()  # uint8 to fp16/32
        img_padded /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_padded.ndimension() == 3:
            img_padded = img_padded.unsqueeze(0)
        # Inference
        pred = self.model(img_padded)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        # Output bounding boxes
        detections = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            log_detection, frame = '', getattr(dataset, 'frame', 0)
            log_detection += '%gx%g ' % img_padded.shape[2:]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_padded.shape[2:], det[:, :4], img_shape).round()
                # Detection results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    log_detection += f"{n} detection{'s' * (n > 1)}"  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy= np.array([xyxy[0].cpu(), xyxy[1].cpu(), xyxy[2].cpu(), xyxy[3].cpu()]).astype(int)
                    name = self.names[int(cls.item())]
                    color = self.colors[name]
                    detection = Detection(frame_n=frame_n, xyxy=xyxy, color=color, name=name, conf=conf.item())
                    detections.append(detection)
        return detections, log_detection