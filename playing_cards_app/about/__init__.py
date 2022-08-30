from flask import Blueprint

bp = Blueprint('about', __name__)

from playing_cards_app.about import routes
