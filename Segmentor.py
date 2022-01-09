import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import numpy as np
import cv2
import time


class Segmentor(object):

	def __init__(self, path, device):
		self.path = path
		self.device = device
		self.model = self.load()

	def get_endpoints(self, segmentation):
		# Try catch to fix
		_, bw = cv2.threshold(segmentation, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		if len(contours) != 0:
			c = max(contours, key=cv2.contourArea)
			data_pts = np.empty((len(c), 2), dtype=np.float64)
			for i in range(data_pts.shape[0]):
				data_pts[i,0] = c[i,0,0]
				data_pts[i,1] = c[i,0,1]
			# Perform PCA analysis
			mean = np.empty((0))
			mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
			ymax, xmax = segmentation.shape[0:2]
			# Center of the object
			center = (int(mean[0,0]), int(mean[0,1]))
			try:
				# First axis of the object
				p1 = (center[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], center[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
				# Second axis of the object
				p2 = (center[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], center[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
				# First axis end points
				p1_first_axis, p2_first_axis = self.get_ext(bw, center, p1, xmax, ymax)
				# Second axis end points
				p1_second_axis, p2_second_axis = self.get_ext(bw, center, p2, xmax, ymax)
			except:
				p1_first_axis = (-1, -1)
				p2_first_axis = (-1, -1)
				p1_second_axis = (-1, -1)
				p2_second_axis = (-1, -1)
				center = (-1, -1)
		else:
			p1_first_axis = (-1, -1)
			p2_first_axis = (-1, -1)
			p1_second_axis = (-1, -1)
			p2_second_axis = (-1, -1)
			center = (-1, -1)
		return (p1_first_axis, p2_first_axis), (p1_second_axis, p2_second_axis), center

	def get_ext(self, bw, p1, p2, xmax, ymax):
		x = np.linspace(0, xmax-1, xmax*10)
		y = self.f(p1, p2, x)
		good = np.where((y>0) & (y<ymax))
		y = y[good]
		x = x[good]
		ext = [(int(y0), int(x0)) for x0, y0 in zip(x, y) if bw[int(y0), int(x0)] == 255]
		if(not(ext)):
			new_p1 = (-1, -1)
			new_p2 = (-1, -1)
		else:
			new_p1 = (ext[0][1], ext[0][0])
			new_p2 = (ext[-1][1] , ext[-1][0])
		return new_p1, new_p2

	def f(self, p1, p2, x):
	    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
	    b = p1[1] - a * p1[0]
	    return a*x + b

	def transform(self, img):
		im = cv2.resize(img, (513, 513), cv2.INTER_CUBIC)
		im = im[:, :, ::-1] # BGR to RGB
		im = np.ascontiguousarray(im, dtype=np.float32)
		im /= 255.0 # 0-255 to 0.-1.
		im -= [0.485, 0.456, 0.406] # ImageNet mean
		im /= [0.229, 0.224, 0.225] # ImageNet std
		im = im.transpose(2, 0, 1)  # h,w,c to c,w,h
		# Line if error 4-dimensional, depend on barch or not
		im = im[np.newaxis,:,:,:]
		return im

	def test(self, outs):
		return [out.argmax(1).cpu().numpy().astype(np.uint8) for out in outs]



	def segment(self, img, detections):
		"""
		t00 = time.time()

		ims = []
		shapes = []


		for detection in detections:
			xyxy = detection.xyxy
			thumbnail = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
			h, w = thumbnail.shape[:2]
			thumbnail = self.transform(thumbnail)
			shapes.append((w, h))
			ims.append(thumbnail)
		"""
		"""
		for detection in detections:
			xyxy = detection.xyxy
			thumbnail = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
			h, w = thumbnail.shape[:2]
			thumbnail = self.transform(thumbnail)
			shapes.append((w, h))
			ims.append(thumbnail)

		times = []
		tmod = []
		tout = []

		u = torch.Tensor(np.array([0])).cuda()

		t0 = time.time()
		for i in range(10):
			t00 = time.time()
			outs = self.model(torch.Tensor(ims).to(self.device))['out']
			t01 = time.time()
			outs = outs.argmax(1).cpu().numpy().astype(np.uint8)
			t02 = time.time()
			tmod.append(t01-t00)
			tout.append(t02-t01)
			times.append(t02-t00)
		"""
		"""
		t1 = time.time()
		
		print(times)
		print(tmod)
		print(tout)

		print((t1-t0)/10)
		print(min(times))
		print(max(times))
		print(min(tmod))
		print(max(tmod))
		print(min(tout))
		print(max(tout))

		print("ok")
		#torch.cuda.synchronize()
		torch.cuda.synchronize()
		t0 = time.time()
		outs = self.model(torch.Tensor(ims).to(self.device))['out']
		#outs = self.model(torch.Tensor(ims).to(self.device))['out']

		torch.cuda.synchronize()
		t1 = time.time()
		#torch.cuda.synchronize()
		u = outs.argmax(1).cpu().numpy().astype(np.uint8)

		torch.cuda.synchronize()
		t2 = time.time()
		u = outs.argmax(1).cpu().numpy().astype(np.uint8)

		torch.cuda.synchronize()
		t2 = time.time()

		torch.cuda.synchronize()
		t3 = time.time()
		print(t1-t0)
		print(t2-t1)
		print(t3-t2)
		print(t3-t0)

		print("ok")
		#torch.cuda.synchronize()
		torch.cuda.synchronize()
		t0 = time.time()
		outs = self.model(torch.Tensor(ims).to(self.device))['out']
		#outs = self.model(torch.Tensor(ims).to(self.device))['out']

		torch.cuda.synchronize()
		t1 = time.time()
		#torch.cuda.synchronize()
		u = outs.argmax(1).cpu().numpy().astype(np.uint8)

		torch.cuda.synchronize()
		t2 = time.time()
		u = outs.argmax(1).cpu().numpy().astype(np.uint8)

		torch.cuda.synchronize()
		t2 = time.time()

		torch.cuda.synchronize()
		t3 = time.time()
		print(t1-t0)
		print(t2-t1)
		print(t3-t2)
		print(t3-t0)


		segs2 = []
		"""

		
		"""
		for i, out in enumerate(outs):
			out = out.argmax(1).cpu().numpy().astype(np.uint8)
			#out = out.cpu().numpy()
			#out = out.astype(np.uint8)
			out = np.squeeze(out) * 255
			#out *= 255
			out = cv2.resize(out, shapes[i], cv2.INTER_CUBIC)
			segs2.append(out)

		#t5 = time.time()
		#print(t5-t4)

		for i, im in enumerate(ims):
			out = self.model(torch.Tensor(ims).to(self.device))['out']
			out = out.argmax(1)
			out = out.cpu().numpy()
			out = out.astype(np.uint8)
			out = np.squeeze(out)
			out *= 255
			out = cv2.resize(out, shapes[i], cv2.INTER_CUBIC)
			outs.append(out)
		"""
		#t2 = time.time()

		#print("Preparation :", t1-t0)
		#print("Segmentation:", t2-t1)
		#print("Total:", t2-t0)

		xyxy = detections.xyxy
		thumbnail = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
		h, w = thumbnail.shape[:2]
		thumbnail = self.transform(thumbnail)
		shapes = (w, h)
		#shapes.append((w, h))
		#ims.append(thumbnail)

		out = self.model(torch.Tensor(thumbnail).to(self.device))['out']
		out = out.argmax(1).cpu().numpy().astype(np.uint8)
		out = np.squeeze(out) * 255
		out = cv2.resize(out, shapes, cv2.INTER_CUBIC)

		return out

	def load(self):
		model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
		model.classifier = DeepLabHead(2048, 2)
		model.load_state_dict(torch.load(self.path))
		model.eval()
		model.to(self.device)
		return model


