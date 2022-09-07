from flask import Flask, request, Response, flash, make_response, current_app, send_from_directory
from flask import render_template, url_for, redirect
from playing_cards_app.demo import bp
from playing_cards_app.demo.utils import gen_frames, get_picture, save_image
from playing_cards_app.demo.predictions import predict, get_saliency, get_readable_concepts
import cv2
from datetime import datetime
import os
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from playing_cards_app.demo.data.utils import IndexToString


@bp.route('/demo', methods=["GET", "POST"])
def demo():
	if request.method == 'GET' or request.form["submit"] == "reset":
		image = None
		name = None
	else:  # POST
		image = request.form["submit"]
		name = take_picture()
	return render_template('demo/demo.html', title='Demo', image=image, name=name)


@bp.route('/upload_file/', methods=['POST'])
def upload_files():
	uploaded_file = request.files['image_file']
	filename = secure_filename(uploaded_file.filename)
	save_image(filename, uploaded_file, file_exts=current_app.config['UPLOAD_EXTENSIONS'], folder=current_app.config['UPLOAD_FOLDER'], type="upload")
	image = "upload"
	name = filename
	return render_template('demo/demo.html', title='Demo', image=image, name=name)


@bp.route('/video_feed')
def video_feed():
	return Response(gen_frames(device=current_app.config['CAMERA']), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/take_picture')
def take_picture():
	img = get_picture(device=current_app.config['CAMERA'])
	dt = datetime.now()
	filename = f"{dt.microsecond}.jpg"
	save_image(filename, img, file_exts=current_app.config['UPLOAD_EXTENSIONS'], folder=current_app.config['UPLOAD_FOLDER'], type="cv2")
	return filename


@bp.route('/get_image/<path:filename>')
def get_image(filename):
	return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@bp.route('/results', methods=['POST'])
def results():
	input_name = request.form["model_input"]
	task_out, concepts = predict(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], input_name), gen_salience=True, input_name=input_name, file_exts=current_app.config['UPLOAD_EXTENSIONS'], folder=current_app.config['UPLOAD_FOLDER'])

	concept_out = get_readable_concepts(concepts)  # Get human readable outputs
	return render_template('demo/results.html', title='Results', input=input_name, task_out=task_out, concepts=concept_out)


@bp.route('/update_results', methods=['POST'])
def update_results():
	global task_out, concepts, input
	input_name = request.form["model_input"]
	# model_input is not included in the new concept vector
	new_concepts = [float(request.form.get(x)) if x != "model_input" else None for x in request.form][1:]
	task_out, new_concepts = predict(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], input_name), concepts=new_concepts, gen_salience=False, input_name=input_name)
	concept_out = get_readable_concepts(new_concepts)

	return render_template('demo/results.html', title='Results', input=input_name, concepts=concept_out, task_out=task_out)
