from flask import Flask, request, Response, flash
from flask import render_template, url_for, redirect
from playing_cards_app.about import bp
import os


@bp.route('/about')
def about():
	examples = os.listdir(os.path.join("playing_cards_app", "static", "examples"))
	return render_template('about/about.html', title='About', examples=examples)
