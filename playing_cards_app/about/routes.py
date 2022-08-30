from flask import Flask, request, Response, flash
from flask import render_template, url_for, redirect
from playing_cards_app.about import bp


@bp.route('/about')
def about():
	return render_template('about/about.html', title='About')
