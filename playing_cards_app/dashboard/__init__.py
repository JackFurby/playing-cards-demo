from flask import Blueprint

bp = Blueprint('dashboard', __name__)

from playing_cards_app.dashboard import routes
from playing_cards_app.dashboard import forms
