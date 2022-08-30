from flask import Flask, request, Response, flash
from flask import render_template, url_for, redirect
from playing_cards_app.demo import bp


@bp.route('/demo')
def demo():
	return render_template('demo/demo.html', title='Demo')
