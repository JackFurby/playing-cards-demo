from flask import Flask
from config import Config
from celery import Celery
from sassutils.wsgi import SassMiddleware
import playing_cards_app.forms

import logging


# https://medium.com/@frassetto.stefano/flask-celery-howto-d106958a15fe
def make_celery(app_name=__name__):
	redis_uri = 'redis://localhost:6379'
	return Celery(app_name, backend=redis_uri, broker=redis_uri)


celery = make_celery()


def init_celery(celery, app):
	celery.conf.update(app.config)
	TaskBase = celery.Task

	class ContectTask(TaskBase):
		def __call__(self, *args, **kwargs):
			with app.app_context():
				return TaskBase.__call__(self, *args, **kwargs)
	celery.Task = ContectTask


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

	# backgroud tasks
	init_celery(celery, app)

	# Dashboard
	from playing_cards_app.dashboard import bp as dashboard_bp
	app.register_blueprint(dashboard_bp)

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
