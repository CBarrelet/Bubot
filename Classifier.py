import cv2
import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from main_utils import read_pickle

def efficientnet_forward(self, inputs):
    """ 
    Calls extract_features to extract features, applies final linear layer, and returns logits. 
    From: https://discuss.pytorch.org/t/efficientnet-output-deep-feature-vector-in-addition-to-logit/75112
    """
    bs = inputs.size(0)
    # Convolution layers
    x = self.extract_features(inputs)
    # Pooling and final linear layer
    x = self._avg_pooling(x)
    FV = x.view(bs, -1)
    h = self._dropout(FV)
    Logit = self._fc(h)
    return (FV, Logit)
EfficientNet.forward.__code__ = efficientnet_forward.__code__


class Classification(object):

    def __init__(self, color=(0,0,0), name=str(''), conf=float(0), feature=np.zeros((1,1)), top5=[]):
        self.color = color
        self.name = name
        self.conf = conf
        self.feature = feature
        self.top5 = top5


class Classifier(object):

    def __init__(self, path, device, names_path):
        self.path = path
        self.device = device
        self.names = read_pickle(names_path)
        self.colors = {name: [np.random.randint(0, 255) for _ in range(3)] for name in self.names}
        self.model = self.load()
        self.activation = nn.Softmax(dim=1)

    def load(self):
        """
        Load EfficientNet model with trained weights.

        Parameters:
            - path: path to the trained weights.
            - device: cuda device, i.e. 0 or 0,1,2,3 or cpu 
        Return:
            - model: 120 classes trained EfficientNet-B6
        """
        model = EfficientNet.from_pretrained(model_name='efficientnet-b6', num_classes=120)
        model.load_state_dict(torch.load(self.path))
        model.to(self.device)
        model.eval()
        return model

    def classify(self, imgs, detections):
        """
        Classify boxes and set detection's classification.

        Parameters:
            - imgs: List[ndarray, ndarray]
                List containing left and right images
        """ 
        ims = []
        for i in [0, 1]:
            for detection in detections[i]:
                bbx = detection.xyxy
                cutout = imgs[i][bbx[1]:bbx[3], bbx[0]:bbx[2]]
                im = self.transform(cutout)
                ims.append(im)
        if(len(ims)):
            features_vector, preds = self.model(torch.Tensor(ims).to(self.device))
            preds = self.activation(preds)
            topk, topclass = preds.topk(5, dim=1)
            classes_ = topclass.cpu().numpy()
            confs_ = topk.cpu().numpy()
            for i, (class_, conf_) in enumerate(zip(classes_, confs_)):
                name = self.names[class_[0]]
                conf = conf_[0]
                feature_vector = features_vector[i].cpu().numpy()
                names = [self.names[class_[i]] for i in range(5)]
                top5 = (names, conf_)
                color = self.colors[name]
                classification = Classification(color=color, name=name, conf=conf, feature=feature_vector, top5=top5)
                if(i < len(detections[0])):
                    detections[0][i].set_classification(classification)
                else:
                    detections[1][i-len(detections[0])].set_classification(classification)

    def transform(self, img):
        """
        Perform various transformations on the image.

        Parameters:
            - img: ndarray BGR image [h,w,c]
                The image to apply the transformation
        Return:
            - im: ndarray RGB image [c,w,h] 
                The transformed image.
        """ 
        im = cv2.resize(img, (224, 224)) 
        im = im[:, :, ::-1] # BGR to RGB
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255.0 # 0-255 to 0.-1.
        im -= [0.485, 0.456, 0.406] # ImageNet mean
        im /= [0.229, 0.224, 0.225] # ImageNet std
        im = im.transpose(2, 0, 1)  # h,w,c to c,w,h
        return im
