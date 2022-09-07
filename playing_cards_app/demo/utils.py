import cv2
import imageio as iio
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from playing_cards_app import celery


def get_camera(device="<video0>"):
	"""cv2 works for IP camera but does not release USB cameras.
	Select libary based on device type
	"""
	print(device)
	if device[:4] == "http":
		camera = cv2.VideoCapture(device)
		cv2_capture = True
	else:
		camera = iio.get_reader(device)
		cv2_capture = False
	return camera, cv2_capture


def gen_frames(device):
	camera, cv2_capture = get_camera(device=device)
	delay = 1 / 30  # 30 fps
	while True:
		frame = get_frame(camera, cv2_capture)
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		time.sleep(delay)
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def get_picture(device):
	time.sleep(0.3)  # make sure the camera is available (may be used by stream before taking picture)
	camera, cv2_capture = get_camera(device=device)
	return get_frame(camera, cv2_capture)


def get_frame(camera, cv2_capture=False):
	if cv2_capture:
		ret, frame = camera.read()
	else:
		frame = camera.get_next_data()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = square_crop(frame)  # discard parts of frame we cannot pass into model
	return frame


def square_crop(img):
	"""
	Apply a square crop to a given image. The crop will also be a center crop
	"""

	height, width, channels = img.shape
	# Get the smallest dim so the retured image is the largest possible for a
	# square crop
	if height > width:
		new_size = width
	else:
		new_size = height
	x = width/2 - new_size/2  # start x coordinate
	y = height/2 - new_size/2  # start y coordinate

	return img[int(y):int(y+new_size), int(x):int(x+new_size)]


@celery.task()
def save_image(filename, img, file_exts, folder, type, cmap="seismic"):
	""" Save a given image
	args:
		filename (string): Name to save image as
		img (cv2 image): image to save
		type (string): type of image to save (options: cv2, upload, plt)
	"""
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in file_exts:
			abort(400)
	if not os.path.exists(f"playing_cards_app/{folder}"):
		os.makedirs(f"playing_cards_app/{folder}")
	if type == "cv2":
		cv2.imwrite(os.path.join("playing_cards_app", folder, filename), img)
	elif type == "upload":
		img.save(os.path.join("playing_cards_app", folder, filename))
	elif type == "plt":
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img, cmap=cmap)
		plt.savefig(os.path.join("playing_cards_app", folder, filename))
		plt.close()
