from setuptools import setup

setup(
    name='playing_cards_app',
    version="0.0.1",
    packages=['playing_cards_app'],
    setup_requires=['libsass >= 0.6.0'],
    sass_manifests={
        'playing_cards_app': ('static/scss/', 'static/css', '/static/css')
    },
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)
