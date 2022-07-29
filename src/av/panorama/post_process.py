import cv2
import numpy as np


def crop(img):
    return img[200:548, 45:1450]

def blend():
    pass

def read_img(img_path: str) -> np.ndarray:
    """Reads an image of a given path."""
    img = cv2.imread(img_path)
    return img


def show_img(img: np.ndarray) -> None:
	"""Displays an np.ndarray image."""
	cv2.imshow("Image", img)
	cv2.waitKey(0)


if __name__ == '__main__':
    path = "/home/norbert/repos/sentrynode/sn-perception/Stitched_Panorama.png"
    img = read_img(path)
    img = crop(img)
    show_img(img)
