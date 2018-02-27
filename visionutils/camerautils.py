# Version 0.0.1
# Last Updated 2018-02-15

__version__ = "0.0.1"
__author__ = "Michael Chatten"

import numpy as np
import cv2, sys, os, time


class CameraUtils:

	def __init__(self):
		self.w = 224
		self.h = 224
		self.bias_w = 'c' # c = center, l = left, r = right
		self.bias_h = 'c' # c = center, t = top, b = bottom
		self.camera = 0
		self.frame_rate = 30.0
		self.video_ext = '.mp4'
		self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		self.cap = None
		self.batch = None	
		self.show_fps = False
		self.fps_time = time.time()
		self.fps_count = 0
		self.fps = 0.0

	def set_crop_bias(self, bias):
		self.bias_h = 'c'
		self.bias_w = 'c'
		if bias[0] == 't': self.bias_h = 't'
		if bias[0] == 'b': self.bias_h = 'b'
		if bias[1] == 'l': self.bias_w = 'l'
		if bias[1] == 'r': self.bias_w = 'r'

	def set_resolution(self, w, h):
		if isinstance(w, int):
			self.w = w
		if isinstance(h, int):
			self.h = h	

	def set_camera(self, id=0):
		self.camera = 0
		if isinstance(id, int):
			self.camera = id

	def load_image(self, filename):
		print(filename)
		return cv2.imread(filename)

	def annotate_image(self, image, notes):
		cnt = 0
		for note in notes:
			cv2.putText(image, note, (10, 20 + 20*cnt), cv2.FONT_HERSHEY_SIMPLEX, .45, (64, 255, 255), 2)
			cnt += 1
		if self.show_fps:
			cv2.putText(image, "{0:.3} fps".format(self.fps), (image.shape[1]-60, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, .35, (64, 255, 255), 1)
			self.fps_count += 1
			if time.time() - self.fps_time > 1:	
				self.fps = self.fps_count / (time.time() - self.fps_time)			
				self.fps_count = 0
				self.fps_time = time.time()
		return image		

	def show_image(self, image, pause=True, notes=[]):
		image = self.annotate_image(image, notes)
		cv2.imshow('image', image)
		if pause:
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			cv2.waitKey(int(self.frame_rate))

	def batch_capture(self, frames, w=-1, h=-1):
		self.batch = []
		for i in range(frames):
			self.batch.append(self.capture_image(w, h))
		return self.batch

	def show_batch(self, images, col=6, row=5, scale=1, notes=[]):
		grid_image = np.array([])
		horz_imgs = []
		for i in range(row):
			horz_imgs.append(np.hstack(list(images[i * col:i * col + col])))
		grid_image = np.vstack(list(horz_imgs))
		if scale != 1:
			grid_image = cv2.resize(grid_image, (int(grid_image.shape[1]*scale), int(grid_image.shape[0]*scale)))

		self.show_image(grid_image, False, notes)

	def capture_image(self, w=-1, h=-1, notes=[]):
		if w == -1: w = self.w
		if h == -1: h = self.h		
		ret, frame = self.cap.read()
		if ret==True:				
			frame = self.annotate_image(frame, notes)
			return True, self.resize_image(frame, w, h)
		return None

	def resize_image(self, image, w=-1, h=-1, log=False):
		if w == -1: w = self.w
		if h == -1: h = self.h

		if log: print(image.shape)
		vh = image.shape[0]
		vw = image.shape[1]
		vr = vw / float(vh)

		# Scale image to pre-crop maximum
		#
		if w > vw or h > vh:
			scale = vw/float(w)
			if vh/float(h) > scale: scale = vh/float(h)
		else:
			scale = w/float(vw)
		if h/float(vh) > scale: scale = h/float(vh)
		image = cv2.resize(image, (int(vw*scale)+1, int(vh*scale)+1))
		vh = image.shape[0]
		vw = image.shape[1]
		vr = vw / float(vh)
		if log: print(image.shape)

		top = 0
		off_h = 0
		if self.bias_h == 'c':
			top = int(max(0, image.shape[0] - h)/2)
		elif self.bias_h == 'b':
			top = int(max(0, image.shape[0] - h))

		left = 0
		if self.bias_w == 'c':
			left = int(max(0, image.shape[1] - w)/2)
		elif self.bias_w == 'r':
			left = int(max(0, image.shape[1] - w))

		# Crop to require width / hieght
		#
		image = image[top: h + top, left: w + left]
		if log: print(image.shape)

		if log: print(image.shape)
		return image

	def save_image(self, image, filename, comp=0):
		self.create_directory_path(filename)
		cv2.imwrite(filename + '.png', image,  [cv2.IMWRITE_PNG_COMPRESSION, comp])		

	def test_camera(self, frames=300):
		self.capture_video('', frames, self.w, self.h, 'none')

	def capture_camera_to_frames(self, filename, frames, w=-1, h=-1):
		self.capture_video(filename, frames, w, h, 'image')

	def attach_camera(self, id=-1):
		if id != -1: self.set_camera(id)
		self.cap = cv2.VideoCapture(self.camera)		

	def detach_camera(self):
		self.cap.release()

	def destroy_windows(self):
		cv2.destroyAllWindows()

	def capture_video(self, filename, frames, w=-1, h=-1, output='video'):
		if w == -1: w = self.w
		if h == -1: h = self.h
		self.create_directory_path(filename)

		self.attach_camera()
		# cap = cv2.VideoCapture(self.camera)		
		if output == 'video':
			out = cv2.VideoWriter()
			succes = out.open(filename + self.video_ext, self.fourcc, self.frame_rate, (w, h), True)

		cnt = 0
		while(self.cap.isOpened()):
			ret, frame = self.cap.read()
			if ret==True:				
				# SCALE FRAME
				frame = self.resize_image(frame, w, h)

				# WRITE IMAGE TO FILE
				if output == 'image':
					self.save_image(frame, filename + '-%05d' % cnt)
					# cv2.imwrite(filename + '-%05d.png' % cnt, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

				# WRITE TO VIDEO FILE
				if output == 'video': 
					out.write(frame)

				# DISPLAY VIDEO
				cv2.imshow('frame', frame)

				cnt += 1
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				if cnt > frames:
					break

		# cap.release()
		self.detach_camera()
		if output == 'video': out.release()
		cv2.destroyAllWindows()

	def convert_video_to_frames(self, video_filename, filename, w=-1, h=-1):		
		vidcap = cv2.VideoCapture(video_filename)
		success,image = vidcap.read()
		cnt = 0

		if w == -1: w = image.shape[1]
		if h == -1: h = image.shape[0]

		while success:
			image = self.resize_image(image, w, h)
			self.save_image(image, filename + '-%05d' % cnt)
			success,image = vidcap.read()
			cnt += 1

	def create_directory_path(self, filename):
		if len(filename.split('/')) > 1:
			os.makedirs(os.path.dirname(filename), exist_ok=True)




