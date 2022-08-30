from flask import Blueprint

bp = Blueprint('demo', __name__)

from playing_cards_app.demo import routes
