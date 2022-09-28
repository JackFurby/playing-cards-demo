import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
	UPLOAD_EXTENSIONS = ['.jpg', '.jpeg', '.png']
	UPLOAD_FOLDER = "uploads"
	CAMERA = os.environ.get('PLAYING_CARDS_DEMO_CAMERA')
	CONCEPT_SORT = "concept"
	MODEL_NAME = os.environ.get('PLAYING_CARDS_DEMO_MODEL')
