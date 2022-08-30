from playing_cards_app import create_app
from config_testing import TestConfig
import pytest


@pytest.fixture(scope='function')
def app_client(db, app):
	"""Create api request client."""
	return app
