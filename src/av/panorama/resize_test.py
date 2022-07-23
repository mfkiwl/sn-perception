import cv2
import numpy as np


def resize_img(cv_img: np.ndarray, width: int, height: int) -> np.ndarray:
	"""Resize an OpenCV image with the given parameters."""
	cv_img = cv2.resize(
		cv_img, (width, height), interpolation=cv2.INTER_AREA
	)
	# cv_img = cv_img[..., np.newaxis]
	return cv_img


def show_img(img_path: str, img_is_resized: bool=False, width: int=None, height: int=None) -> None:
	"""Displays an image of a given path."""
	img = cv2.imread(img_path)
	if img_is_resized:
		img = resize_img(img.copy(), width, height)
	cv2.imshow("Image", img)
	cv2.waitKey(0)


if __name__ == '__main__':
	IMG_DIR: str = "src/av/panorama/assets/indoors/room"
	FRAME_WIDTH: int = 680
	FRAME_HEIGHT: int = 384

	for i in range(3):
		# without resizing
		#show_img(IMG_DIR + "/" + f"{i + 1}.jpg")

		# with resizing
		show_img(IMG_DIR + "/" + f"{i + 1}.jpg", True, FRAME_WIDTH, FRAME_HEIGHT)
