import copy
import cv2
import threading
import queue
import numpy

from typing import Optional
from vimba import *

# extra imports
from pano_class import CylindricalStitcher

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 680
FOCAL_LENGTH = 540

def resize_img(cv_img: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
	"""Resize an OpenCV image with the given parameters."""
	cv_img = cv2.resize(
		cv_img, (width, height), interpolation=cv2.INTER_AREA
	)
	# cv_img = cv_img[..., numpy.newaxis]
	return cv_img


def add_camera_id(frame: Frame, cam_id: str) -> Frame:
	# Helper function inserting 'cam_id' into given frame. This function
	# manipulates the original image buffer inside frame object.
	cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
				color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
	return frame


def resize_if_required(frame: Frame) -> numpy.ndarray:
	# Helper function resizing the given frame, if it has not the required dimensions.
	# On resizing, the image data is copied and resized, the image inside the frame object
	# is untouched.
	cv_frame = frame.as_opencv_image()

	if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
		cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
		cv_frame = cv_frame[..., numpy.newaxis]

	return cv_frame


def create_dummy_frame() -> numpy.ndarray:
	cv_frame = numpy.zeros((384, 680, 3), numpy.uint8)
	cv_frame[:] = 0

	cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
				fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

	return cv_frame


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
	try:
		q.put_nowait((cam.get_id(), frame))

	except queue.Full:
		pass


def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
	# Helper function that tries to set a given value. If setting of the initial value failed
	# it calculates the nearest valid value and sets the result. This function is intended to
	# be used with Height and Width Features because not all Cameras allow the same values
	# for height and width.
	feat = cam.get_feature_by_name(feat_name)

	try:
		feat.set(feat_value)

	except VimbaFeatureError:
		min_, max_ = feat.get_range()
		inc = feat.get_increment()

		if feat_value <= min_:
			val = min_

		elif feat_value >= max_:
			val = max_

		else:
			val = (((feat_value - min_) // inc) * inc) + min_

		feat.set(val)

		msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
			   'Using nearest valid value \'{}\'. Note that, this causes resizing '
			   'during processing, reducing the frame rate.')
		Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))


# Thread Objects
class FrameProducer(threading.Thread):
	def __init__(self, cam: Camera, frame_queue: queue.Queue):
		threading.Thread.__init__(self)

		self.log = Log.get_instance()
		self.cam = cam
		self.frame_queue = frame_queue
		self.killswitch = threading.Event()

	def __call__(self, cam: Camera, frame: Frame):
		# This method is executed within VimbaC context. All incoming frames
		# are reused for later frame acquisition. If a frame shall be queued, the
		# frame must be copied and the copy must be sent, otherwise the acquired
		# frame will be overridden as soon as the frame is reused.
		if frame.get_status() == FrameStatus.Complete:

			if not self.frame_queue.full():
				frame_cpy = copy.deepcopy(frame)
				try_put_frame(self.frame_queue, cam, frame_cpy)

		cam.queue_frame(frame)

	def stop(self):
		self.killswitch.set()

	def setup_camera(self):
		set_nearest_value(self.cam, 'Width', FRAME_WIDTH)
		set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)

		# Enable white balancing
		try:
			self.cam.BalanceWhiteAuto.set('Continuous')

		except (AttributeError, VimbaFeatureError):
			self.log.info('Camera {}: Failed to set Feature \'BalanceWhiteAuto\'.'.format(
						  self.cam.get_id()))

		# Try to enable automatic exposure time setting
		try:
			self.cam.ExposureAuto.set('Once')

		except (AttributeError, VimbaFeatureError):
			self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
						  self.cam.get_id()))

		self.cam.set_pixel_format(PixelFormat.Bgr8)

	def run(self):
		self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))

		try:
			with self.cam:
				self.setup_camera()

				try:
					self.cam.start_streaming(self)
					self.killswitch.wait()

				finally:
					self.cam.stop_streaming()

		except VimbaCameraError:
			pass

		finally:
			try_put_frame(self.frame_queue, self.cam, None)

		self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))


class FrameConsumer(threading.Thread):
	def __init__(self, frame_queue: queue.Queue):
		threading.Thread.__init__(self)

		self.log = Log.get_instance()
		self.frame_queue = frame_queue
		self.stitcher = CylindricalStitcher(IMAGE_HEIGHT, IMAGE_WIDTH, FOCAL_LENGTH)
		self.frame_left = create_dummy_frame()
		self.frame_middle = create_dummy_frame()
		self.frame_right = create_dummy_frame()

		self.result = create_dummy_frame()


	def frame_processing(self, frames):
		"""
		Main frame processing function.
		"""

		convert = lambda text: int(text) if text.isdigit() else text.lower()
		cv_images = [
			resize_img(frames[cam_id].as_numpy_ndarray(), IMAGE_WIDTH, IMAGE_HEIGHT) for cam_id in sorted(frames.keys(), 
				key=lambda x: ([str,int].index( type(convert(x[-1])) ), x)
			)
		]

		if len(cv_images) == 0:
			pass
		elif len(cv_images) == 1:
			self.frame_left, self.frame_middle, self.frame_right = cv_images[0], create_dummy_frame(), create_dummy_frame()
		elif len(cv_images) == 2:
			self.frame_left, self.frame_middle, self.frame_right = cv_images[0], cv_images[1], create_dummy_frame()
		else:
			self.frame_left, self.frame_middle, self.frame_right = cv_images[0], cv_images[1], cv_images[2]

		# convert to 3-channel
		# classframes = [self.frame_left, self.frame_middle, self.frame_right]
		# classframes = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in classframes]

		self.result = self.stitcher.create_panorama(
			[ self.frame_middle, self.frame_left, self.frame_right ]
			)

		# self.result = numpy.concatenate([self.frame_left, self.frame_middle, self.frame_right], axis=1)

		cv2.imshow('Multithreading Example: Press <Enter> to exit', self.result)

		# try:
		# 	self.result = self.stitcher.create_panorama(
		# 		[ cv_images[1], cv_images[0], cv_images[2] ]
		# 	)
		# except:
		# 	pass
		# cv2.imshow('Multithreading Example: Press <Enter> to exit', cv2.resize(self.result, (1280, 960)))


	def run(self):
		IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
		KEY_CODE_ENTER = 13

		frames = {}
		alive = True

		self.log.info('Thread \'FrameConsumer\' started.')

		while alive:
			# Update current state by dequeuing all currently available frames.
			frames_left = self.frame_queue.qsize()
			while frames_left:
				try:
					cam_id, frame = self.frame_queue.get_nowait()

				except queue.Empty:
					break

				# Add/Remove frame from current state.
				if frame:
					frames[cam_id] = frame

				else:
					frames.pop(cam_id, None)

				frames_left -= 1

			# Construct image by stitching frames together.
			if frames:

				self.frame_processing(frames)

				# # code comes here (import code here from another script?)

				# #print(frames.keys())

				# ###
				# cv_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys(), reverse=True)]
				
				# try:
				#     self.result = self.stitcher.stitch([cv_images[0], cv_images[1]])
				# except:
				#     pass
				
				# cv2.imshow('Result', self.result)               

			# If there are no frames available, show dummy image instead
			else:
				cv2.imshow(IMAGE_CAPTION, create_dummy_frame())


			# Check for shutdown condition
			if KEY_CODE_ENTER == cv2.waitKey(10):
				cv2.destroyAllWindows()
				alive = False

		self.log.info('Thread \'FrameConsumer\' terminated.')


class MainThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

		self.FRAME_QUEUE_SIZE = 3
		self.frame_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
		self.producers = {}
		self.producers_lock = threading.Lock()

	def __call__(self, cam: Camera, event: CameraEvent):
		# New camera was detected. Create FrameProducer, add it to active FrameProducers
		if event == CameraEvent.Detected:
			with self.producers_lock:
				self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
				self.producers[cam.get_id()].start()

		# An existing camera was disconnected, stop associated FrameProducer.
		elif event == CameraEvent.Missing:
			with self.producers_lock:
				producer = self.producers.pop(cam.get_id())
				producer.stop()
				producer.join()

	def run(self):
		log = Log.get_instance()
		consumer = FrameConsumer(self.frame_queue)

		vimba = Vimba.get_instance()
		vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

		log.info('Thread \'MainThread\' started.')

		with vimba:
			# Construct FrameProducer threads for all detected cameras
			for cam in vimba.get_all_cameras():
				self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)

			# Start FrameProducer threads
			with self.producers_lock:
				for producer in self.producers.values():
					producer.start()

			# Start and wait for consumer to terminate
			vimba.register_camera_change_handler(self)
			consumer.start()
			consumer.join()
			vimba.unregister_camera_change_handler(self)

			# Stop all FrameProducer threads
			with self.producers_lock:
				# Initiate concurrent shutdown
				for producer in self.producers.values():
					producer.stop()

				# Wait for shutdown to complete
				for producer in self.producers.values():
					producer.join()

		log.info('Thread \'MainThread\' terminated.')


if __name__ == '__main__':
	main = MainThread()
	main.start()
	main.join()