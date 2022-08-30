from flask import Flask, request, Response, flash
from flask import render_template, url_for, redirect
from playing_cards_app.dashboard import bp


@bp.route('/')
@bp.route('/dashboard')
def dashboard():
	return render_template('dashboard/dashboard.html', title='Home')
