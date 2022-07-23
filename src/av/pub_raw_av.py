import threading
import time
import copy
from typing import Optional

# A multi-producer, multi-consumer queue.
import queue 

from vimba import (
	Vimba, Frame, FrameStatus, Log, PixelFormat,
	VimbaCameraError, VimbaFeatureError
)
from vimba.camera import Camera, CameraEvent
from vimba.util.log import LOG_CONFIG_INFO_CONSOLE_ONLY

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np


#TODO: Parameters as arguments.
FRAME_HEIGHT = 384
FRAME_WIDTH = 680
IMAGE_CAPTION = 'Panorama Image: Press <Enter> to exit'
KEY_CODE_ENTER = 13


#TODO: Put helper functions into a separate package.
def create_dummy_frame() -> np.ndarray:
	"""
	Creates placeholder dummy frame when no image is available.
	"""
	cv_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 1), np.uint8)
	
	cv_frame[:] = 0
	cv2.putText(
		cv_frame, 'No Stream available. Please connect a Camera.', 
		org=(30, 30), fontScale=1, color=255, thickness=1,
		fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL
	)
	return cv_frame


def resize_if_required(frame: Frame) -> np.ndarray:
	"""
	Helper function resizing the given frame, if it 
	does not have the required dimensions. On resizing,
	the image data is copied and resized, the image inside the frame object
	is untouched.
	"""
	cv_frame = frame.as_opencv_image()

	if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
		cv_frame = cv2.resize(
			cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA
		)
		#cv_frame = cv_frame[..., np.newaxis]

	return cv_frame


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
	try:
		q.put_nowait((cam.get_id(), frame))

	except queue.Full:
		pass

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
	"""
	Helper function that tries to set a given value. If setting of the initial value failed
	it calculates the nearest valid value and sets the result. This function is intended to
	be used with Height and Width Features because not all Cameras allow the same values
	for height and width.
	"""
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


class VimbaPublisher(Node):

	def __init__(self):
		super().__init__('vimba_publisher')
		
		qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

		self.publisher_left_ = self.create_publisher(
			Image, 'vision/cam/front/vimba_left', 10,
		)
		self.publisher_middle_ = self.create_publisher(
			Image, 'vision/cam/front/vimba_middle', 10, 
		)
		self.publisher_right_ = self.create_publisher(
			Image, 'vision/cam/front/vimba_right', 10, # qos_profile=qos_profile
		)
		self.img_left = None
		self.img_middle = None
		self.img_right = None

		timer_period = 1.0
		self.timer = self.create_timer(timer_period, self.timer_callback)
		self.bridge = CvBridge()
	   
	def timer_callback(self): 
	
		img_msg_left = self.bridge.cv2_to_imgmsg(
			np.uint8(self.img_left), encoding='passthrough'
		)	
		img_msg_left.header = Header(
					frame_id="map",
					stamp=self.get_clock().now().to_msg()
				)
		img_msg_middle = self.bridge.cv2_to_imgmsg(
			np.uint8(self.img_middle), encoding='passthrough'
		)	
		img_msg_middle.header = Header(
					frame_id="map",
					stamp=self.get_clock().now().to_msg()
				)
		img_msg_right = self.bridge.cv2_to_imgmsg(
			np.uint8(self.img_right), encoding='passthrough'
		)	
		img_msg_right.header = Header(
					frame_id="map",
					stamp=self.get_clock().now().to_msg()
				)

		self.publisher_left_.publish(img_msg_left)
		self.publisher_middle_.publish(img_msg_middle)
		self.publisher_right_.publish(img_msg_right)

		self.get_logger().info('[INFO] Publishing...')


class FrameProducerThread(threading.Thread):
	def __init__(self, cam: Camera, frame_queue: queue.Queue):
		threading.Thread.__init__(self)

		self.log = Log.get_instance()
		self.cam = cam
		self.frame_queue = frame_queue
		self.killswitch = threading.Event()

	def __call__(self, cam: Camera, frame: Frame):
		"""
		This method is executed within VimbaC context. All incoming frames
		are reused for later frame acquisition. If a frame shall be queued, the
		frame must be copied and the copy must be sent, otherwise the acquired
		frame will be overridden as soon as the frame is reused.
		"""
		if frame.get_status() == FrameStatus.Complete:

			if not self.frame_queue.full():
				frame_cpy = copy.deepcopy(frame)
				try_put_frame(self.frame_queue, cam, frame_cpy)

		cam.queue_frame(frame)

	def stop(self):
		self.killswitch.set()

	#TODO: Customize camera setup!
	def setup_camera(self):
		set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
		set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

		# Try to enable automatic exposure time setting
		try:
			self.cam.ExposureAuto.set('Once')

		except (AttributeError, VimbaFeatureError):
			self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
						  self.cam.get_id()))

		self.cam.set_pixel_format(PixelFormat.Bgr8)


	def run(self):
		self.log.info(f'[INFO] Producer Thread {self.cam.get_id()} started.')
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


class FrameConsumerThread(threading.Thread):
	def __init__(self, frame_queue: queue.Queue):
		threading.Thread.__init__(self)

		self.log = Log.get_instance()
		self.frame_queue = frame_queue
		
		# init ROS2 to publish when Consumer starts
		rclpy.init(args=None)

		self.publisher = VimbaPublisher()
		self.frame_left = create_dummy_frame()
		self.frame_middle = create_dummy_frame()
		self.frame_right = create_dummy_frame()


	# frame processing function goes here
	def process_frame(self, frames):
		"""
		Main frame processing function. Put the incoming frames
		into variables in order for publishing.
		"""
		# camera order sorting
		convert = lambda text: int(text) if text.isdigit() else text.lower()
		# comprehension list to acquire frames in cv2 (numpy) format
		cv_images = [
			resize_if_required(
				frames[cam_id]) for cam_id in sorted(frames.keys(), 
				key = lambda x: ([str,int].index( type(convert(x[-1])) ), x)
			)
		]
		try:
			self.frame_left, self.frame_middle, self.frame_right = \
				cv_images[0], cv_images[1], cv_images[2]
		except:
			pass

		return self.frame_left, self.frame_middle, self.frame_right

	def run(self):
		
		frames = {}
		alive = True
		self.log.info('[INFO] Consumer Thread started.')

		while alive:
			# Update current state by dequeuing all currently available frames.
			frames_left = self.frame_queue.qsize()
			while frames_left:
				try:
					cam_id, frame = self.frame_queue.get_nowait()
				except queue.Empty:
					break	
				# Add / Remove frame from current state.
				if frame:
					frames[cam_id] = frame
				else:
					frames.pop(cam_id, None)
				frames_left -= 1

			if frames:
				frame_left, frame_middle, frame_right = self.process_frame(frames)
				
				self.publisher.img_left = frame_left.copy()
				self.publisher.img_middle = frame_middle.copy()
				self.publisher.img_right = frame_right.copy()

				rclpy.spin_once(self.publisher)
	
			else:
				pass

		self.log.info('[INFO] Consumer Thread terminated.')


class MainThread(threading.Thread):
	"""
	Class instance is called (__call__) when a new 
	camera is detected. Creates FrameProducer, 
	then adds it to active FrameProducers.
	"""
	def __init__(self):
		threading.Thread.__init__(self)
	
		self.FRAME_QUEUE_SIZE = 10
		self.frame_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
		self.producers = {}
		self.producers_lock = threading.Lock()

	def __call__(self, cam: Camera, event: CameraEvent):
		
		# Set up connected camera during runtime
		if event == CameraEvent.Detected:
			with self.producers_lock:
				self.producers[cam.get_id()] = FrameProducerThread(cam, self.frame_queue)
				self.producers[cam.get_id()].start()
		
		# Shut down disconnected camera's FrameProducer
		elif event == CameraEvent.Missing:
			with self.producers_lock:
				producer = self.producers.pop(cam.get_id())
				producer.stop()
				producer.join()

	def run(self):
		log = Log.get_instance()

		# Construct unified Consumer Thread
		consumer = FrameConsumerThread(self.frame_queue)

		vimba = Vimba.get_instance()
		vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

		log.info('[INFO] Main Thread started.')

		with vimba:
			
			# Construct Producer Threads for all detected cameras
			for cam in vimba.get_all_cameras():
				self.producers[cam.get_id()] = FrameProducerThread(cam, self.frame_queue)

			# Start Producer Threads
			with self.producers_lock:
				for producer in self.producers.values():
					producer.start()

			# Start Consumer Thread
			# THIS SEGMENT RUNS UNTIL TERMINATION 
			vimba.register_camera_change_handler(self)
			consumer.start()
			consumer.join()
			vimba.unregister_camera_change_handler(self)

			# Initiate concurrent shutdown
			with self.producers_lock:
				for producer in self.producers.values():
					producer.stop()

				# Wait for shutdown to complete
				for producer in self.producers.values():
					producer.join()

		log.info('[INFO] Main Thread terminated.')

def main():
	"""
	This is our entry function into the ROS2 node.
	"""
	thr_main = MainThread()
	thr_main.start()
	thr_main.join()

	#print(f'Thread count: {threading.active_count()}')

	# shut down ROS2
	rclpy.shutdown()

if __name__ == '__main__':
	main()