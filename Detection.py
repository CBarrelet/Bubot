import numpy as np
from tracking.deep_sort.detection import Detection as Deep_Detection


class Detection(Deep_Detection):

    def __init__(self, frame_n=-1, xyxy=np.zeros((4,1), dtype=int), color=(0,0,0), name=str(''), conf=float(0), feature=np.zeros((1,1))):
        self.id = 0
        self.frame_n = frame_n

        self.name = name
        self.xyxy = np.array(xyxy, dtype=int)
        tlwh = self.xyxy_to_tlwh()
        self.conf = conf
        super().__init__(tlwh, conf, feature)

        self.classification = None
        self.color = color

        self.is_stereo = False
        self.stereo_detection = None
        self.stero_color = None
        self.measurement = None

        self.segmentation = None

        # Endpoints found by ACP
        self.p1_width = None
        self.p2_width = None
        self.p1_height = None
        self.p2_height = None
        self.center = None

        # Paired endpoints from self endpoints to stereo segmentation
        self.p1_width_12 = None
        self.p2_width_12 = None
        self.p1_height_12 = None
        self.p2_height_12 = None

        # Barycenter of p1 and p2 for left-to-right and right-to-left
        self.p1_width_bary = None
        self.p2_width_bary = None
        self.p1_height_bary = None
        self.p2_height_bary = None
        self.center_2D = None

    def set_endpoints(self, axis_1, axis_2, center):
        if(axis_1[0] != (-1, -1)) and (axis_1[1] != (-1, -1)):
            axis_1 = ((axis_1[0][0]+self.xyxy[0], axis_1[0][1]+self.xyxy[1]), (axis_1[1][0]+self.xyxy[0], axis_1[1][1]+self.xyxy[1]))
        
        if(axis_2[0] != (-1, -1)) and (axis_2[1] != (-1, -1)):
            axis_2 = ((axis_2[0][0]+self.xyxy[0], axis_2[0][1]+self.xyxy[1]), (axis_2[1][0]+self.xyxy[0], axis_2[1][1]+self.xyxy[1]))
        
        if(center != (-1, -1)):
            center = (center[0]+self.xyxy[0], center[1]+self.xyxy[1])
        self.p1_width = axis_1[0]
        self.p2_width = axis_1[1]
        self.p1_height = axis_2[0]
        self.p2_height = axis_2[1]
        self.center = center


    def set_classification(self, classification):
        self.classification = classification
        self.feature = classification.feature

    def set_stereo(self, detection, color):
        self.is_stereo = True
        self.stereo_detection = detection
        self.stereo_color = color

    def set_measurement(self, measurement):
        self.measurement = measurement

    def xyxy_to_tlwh(self):
        ret = self.xyxy.copy()
        ret[2] = abs(self.xyxy[0] - self.xyxy[2])
        ret[3] = abs(self.xyxy[1] - self.xyxy[3])
        return ret

    def set_feature_to_cpu(self):
        self.feature = self.feature.cpu().numpy()