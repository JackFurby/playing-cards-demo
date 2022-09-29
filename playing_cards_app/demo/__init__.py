from flask import Blueprint, current_app
import os
import glob
from config import Config  # this is probably a bad way of doing things

bp = Blueprint('demo', __name__)

from playing_cards_app.demo import routes

# Clears the uploads folder on start-up
uploads_folder = glob.glob(f"{os.path.join('playing_cards_app', Config.UPLOAD_FOLDER)}/*")
for file in uploads_folder:
	os.remove(file)
