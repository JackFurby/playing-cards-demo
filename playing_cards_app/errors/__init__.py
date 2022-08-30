from flask import Blueprint

bp = Blueprint('errors', __name__)

from playing_cards_app.errors import routes
