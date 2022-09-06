from flask import Flask, request, Response, flash, make_response, current_app, send_from_directory
from flask import render_template, url_for, redirect
from playing_cards_app.demo import bp
from playing_cards_app.demo.utils import gen_frames, get_picture, predict, gat_saliency
import cv2
from datetime import datetime
import os
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from playing_cards_app.demo.data.utils import IndexToString


concept_index_to_string = IndexToString("./playing_cards_app/demo/data/concepts.txt")


@bp.route('/demo', methods=["GET", "POST"])
def demo():
	if request.method == 'GET' or request.form["submit"] == "reset":
		image = None
		name = None
	else:
		image = request.form["submit"]
		name = take_picture()
	return render_template('demo/demo.html', title='Demo', image=image, name=name)


@bp.route('/upload_file/', methods=['POST'])
def upload_files():
	uploaded_file = request.files['image_file']
	filename = secure_filename(uploaded_file.filename)
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
			abort(400)
		if not os.path.exists(f"playing_cards_app/{current_app.config['UPLOAD_FOLDER']}"):
			os.makedirs(f"playing_cards_app/{current_app.config['UPLOAD_FOLDER']}")
		uploaded_file.save(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], filename))
	image = "upload"
	name = filename
	return render_template('demo/demo.html', title='Demo', image=image, name=name)


@bp.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/take_picture')
def take_picture():
	img = get_picture()
	dt = datetime.now()
	filename = f"{dt.microsecond}.jpg"
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
			abort(400)
	if not os.path.exists(f"playing_cards_app/{current_app.config['UPLOAD_FOLDER']}"):
		os.makedirs(f"playing_cards_app/{current_app.config['UPLOAD_FOLDER']}")
	cv2.imwrite(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], filename), img)
	return filename


@bp.route('/uploads/<path:filename>')
def upload(filename):
	return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@bp.route('/results', methods=['POST'])
def results():
	global task_out, concepts, input
	input_name = request.form["model_input"]
	task_out, concepts, input = predict(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], input_name))

	# Get human readable outputs
	outputs = []
	for out in [concepts]:
		outputs.append(out)
	sig_concepts = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
	concept_out = []
	for idx, i in enumerate(sig_concepts[0]):
		concept_out.append((idx, concept_index_to_string(idx), i.item()))
	return render_template('demo/results.html', title='Results', input=input_name, task_out=task_out, concepts=concept_out)


@bp.route('/update_results', methods=['POST'])
def update_results():
	global task_out, concepts, input
	input_name = request.form["model_input"]
	# model_input is not included in the new concept vector
	new_concepts = [float(request.form.get(x)) if x != "model_input" else None for x in request.form][1:]
	task_out, new_concepts, input = predict(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], input_name), concepts=new_concepts)

	# Get human readable outputs
	concept_out = []
	for idx, i in enumerate(new_concepts[0]):
		concept_out.append((idx, concept_index_to_string(idx), i.item()))


	concepts = new_concepts
	return render_template('demo/results.html', title='Results', input=input_name, concepts=new_concepts, task_out=task_out)


@bp.route('/saliency', methods=['GET'])
def saliency():
	global task_out, concepts, input

	target_id = int(request.args.get("target"))
	saliency_img = gat_saliency(concepts, input, target_id)

	plt.xticks([])
	plt.yticks([])
	plt.imshow(saliency_img.squeeze(), cmap='seismic')

	# upload saliency map as image
	dt = datetime.now()
	filename = f"{dt.microsecond}.png"
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
			abort(400)
		if not os.path.exists(f"playing_cards_app/{current_app.config['UPLOAD_FOLDER']}"):
			os.makedirs(f"playing_cards_app/{current_app.config['UPLOAD_FOLDER']}")
		plt.savefig(os.path.join("playing_cards_app", current_app.config['UPLOAD_FOLDER'], filename))
	plt.close()
	return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
