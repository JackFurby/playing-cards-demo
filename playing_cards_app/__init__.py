from flask import Flask
from config import Config
from sassutils.wsgi import SassMiddleware
import playing_cards_app.forms

import logging




def create_app(config_class=Config):
	"""
	Construct Flash application without a global variable. This make it easier
	to run unit tests
	"""
	app = Flask(__name__)
	# Add SCSS to CSS while in development
	app.wsgi_app = SassMiddleware(app.wsgi_app, {
		'playing_cards_app': ('static/scss', 'static/css', '/static/css')
	})
	app.config.from_object(config_class)

	# Demo
	from playing_cards_app.demo import bp as demo_bp
	app.register_blueprint(demo_bp)

	# About
	from playing_cards_app.about import bp as about_bp
	app.register_blueprint(about_bp)

	# Error pages and functions
	from playing_cards_app.errors import bp as errors_bp
	app.register_blueprint(errors_bp)

	# Normal app startup
	if not app.debug and not app.testing:
		pass

	return app
