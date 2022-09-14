from flask import Flask, request, Response, flash, make_response, current_app, send_from_directory
from flask import render_template, url_for, redirect
from playing_cards_app.demo import bp
from playing_cards_app.demo.utils import gen_frames, get_picture, save_image
from playing_cards_app.demo.predictions import Model
import cv2
from datetime import datetime
import os
from werkzeug.utils import secure_filename


CURRENT_MODEL = Model()


@bp.route('/demo', methods=["GET", "POST"])
def demo():
	if request.method == 'GET' or request.form["submit"] == "reset":
		image = None
		name = None
		examples = os.listdir(os.path.join("playing_cards_app", "static", "examples"))
	else:  # POST
		image = request.form["submit"]
		name = take_picture()
		examples = None
	return render_template('demo/demo.html', title='Demo', image=image, name=name, examples=examples)


@bp.route('/upload_file/', methods=['POST'])
def upload_files():
	# Selecting examples from examples folder
	if "image_file" in request.form and isinstance(request.form["image_file"], str):
		uploaded_file = cv2.imread(f"{os.path.join('playing_cards_app', 'static', 'examples',request.form['image_file'])}")
		print(f"{os.path.join('playing_cards_app', 'static', request.form['image_file'])}")
		filename = secure_filename(request.form["image_file"])
		save_image(filename, uploaded_file, file_exts=current_app.config['UPLOAD_EXTENSIONS'], folder=current_app.config['UPLOAD_FOLDER'], type="cv2")
	# uploading file from users device
	else:
		uploaded_file = request.files['image_file']
		print(type(uploaded_file))
		filename = secure_filename(uploaded_file.filename)
		save_image(filename, uploaded_file, file_exts=current_app.config['UPLOAD_EXTENSIONS'], folder=current_app.config['UPLOAD_FOLDER'], type="upload")
	image = "upload"
	name = filename
	return render_template('demo/demo.html', title='Demo', image=image, name=name, examples=None)


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
	CURRENT_MODEL.set_input(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], input_name))
	CURRENT_MODEL.predict()
	task_out, task_desc, concept_out, indexed_concepts = CURRENT_MODEL.get_results(sort=current_app.config['CONCEPT_SORT'])

	return render_template('demo/results.html', title='Results', input=input_name, task_out=task_out, task_desc=task_desc, concepts=concept_out, indexed_concepts=indexed_concepts)


@bp.route('/update_results', methods=['POST'])
def update_results():
	# model_input is not included in the new concept vector
	new_concepts = [float(request.form.get(x)) if x != "model_input" else None for x in request.form][1:]
	CURRENT_MODEL.predict(concepts=new_concepts)
	task_out, task_desc, concept_out, indexed_concepts = CURRENT_MODEL.get_results(sort=current_app.config['CONCEPT_SORT'])

	return render_template('demo/results.html', title='Results', input=CURRENT_MODEL.input_name, concepts=concept_out, task_desc=task_desc, task_out=task_out, indexed_concepts=indexed_concepts)


@bp.route('/saliency', methods=['GET'])
def saliency():

	target_id = int(request.args.get("target"))
	filename = CURRENT_MODEL.gen_saliency(target_id, file_exts=current_app.config['UPLOAD_EXTENSIONS'], folder=current_app.config['UPLOAD_FOLDER'])

	return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
