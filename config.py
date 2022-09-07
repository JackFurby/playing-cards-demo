import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
	UPLOAD_EXTENSIONS = ['.jpg', '.png']
	UPLOAD_FOLDER = "uploads"
	CAMERA = "http://192.168.0.5/mjpg/video.mjpg"
