import cv2
import imageio as iio
import time

import torch
from playing_cards_app.demo.data.utils import IndexToString, Transform
from playing_cards_app.demo.model import End2EndModel
from playing_cards_app.demo.lrp_converter import convert
import playing_cards_app.demo.lrp as lrp
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from playing_cards_app import celery

# Global variables and other bits for model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	print("Device:", device, torch.cuda.get_device_name(0))
else:
	print("Device:", device)

NUM_IMG_OUTPUT = 1
USE_SIG = True
N_CONCEPT = 52
N_CLASSES = 6
concept_index_to_string = IndexToString("./playing_cards_app/demo/data/concepts.txt")
class_index_to_string = IndexToString("./playing_cards_app/demo/data/classes.txt")
img_transform = Transform()
XtoCtoY = End2EndModel(num_classes=N_CLASSES, num_concepts=N_CONCEPT, num_images_per_output=1, use_sigmoid=USE_SIG, use_modulelist=False)
XtoCtoY.load_state_dict(torch.load("./playing_cards_app/demo/model_saves/XtoCtoY_converted.pth", map_location=torch.device('cpu')))
XtoC = convert(XtoCtoY.x_to_c_model).to(device)
CtoY = XtoCtoY.c_to_y_model.to(device)

XtoC_model_1 = []
XtoC_model_2 = []
XtoC_model_3 = []

for idx, m in enumerate(XtoC.children()):
	if idx < 15:
		XtoC_model_1.append(m)
	elif idx < 29:
		XtoC_model_2.append(m)
	else:
		XtoC_model_3.append(m)

XtoC_model_1 = lrp.Sequential(*XtoC_model_1)
XtoC_model_2 = lrp.Sequential(*XtoC_model_2)
XtoC_model_3 = lrp.Sequential(*XtoC_model_3)

XtoC_model_1.to(device)
XtoC_model_2.to(device)
XtoC_model_3.to(device)

DEVICE = "http://192.168.0.5/mjpg/video.mjpg"


def get_camera(device="<video0>"):
	"""cv2 works for IP camera but does not release USB cameras.
	Select libary based on device type
	"""
	if device[:4] == "http":
		camera = cv2.VideoCapture(device)
		cv2_capture = True
	else:
		camera = iio.get_reader(device)
		cv2_capture = False
	return camera, cv2_capture


def gen_frames():
	camera, cv2_capture = get_camera(device=DEVICE)
	delay = 1 / 30  # 30 fps
	while True:
		frame = get_frame(camera, cv2_capture)
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		time.sleep(delay)
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def get_picture():
	time.sleep(0.3)  # make sure the camera is available (may be used by stream before)
	camera, cv2_capture = get_camera(device=DEVICE)
	return get_frame(camera, cv2_capture)


def get_frame(camera, cv2_capture=False):
	if cv2_capture:
		ret, frame = camera.read()
	else:
		frame = camera.get_next_data()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = square_crop(frame)  # discard parts of frame we cannot pass into model
	return frame


def square_crop(img):
	"""
	Apply a square crop to a given image. The crop will also be a center crop
	"""

	height, width, channels = img.shape
	# Get the smallest dim so the retured image is the largest possible for a
	# square crop
	if height > width:
		new_size = width
	else:
		new_size = height
	x = width/2 - new_size/2  # start x coordinate
	y = height/2 - new_size/2  # start y coordinate

	return img[int(y):int(y+new_size), int(x):int(x+new_size)]


def predict(image, concepts=None):

	# C to Y only
	if concepts != None:
		c_to_y_input = torch.FloatTensor(concepts).unsqueeze(0)
	# end to end
	else:
		image = Image.open(image).convert('RGBA')
		background = Image.new('RGBA', image.size, (255,255,255))
		image = Image.alpha_composite(background, image)
		image = image.convert('RGB')  # ensure image does not have an alpha channel
		image = img_transform(image).to(device)
		#image, label = images.to(device), labels.to(device)
		image = image.unsqueeze(0)
		image.requires_grad_(True)
		image.grad = None  # Reset gradient

		output_1 = XtoC_model_1.forward(image, explain=True, rule="alpha1beta0")
		output_2 = XtoC_model_2.forward(output_1, explain=True, rule="epsilon")
		pred_concepts = XtoC_model_3.forward(output_2, explain=True, rule="gradient")

		outputs = []
		for out in [pred_concepts]:
			outputs.append(out)
		if USE_SIG:
			c_to_y_input = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
			sig_concepts = c_to_y_input
		else:
			c_to_y_input = torch.cat((outputs), dim=1)
			sig_concepts = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

	task_out = CtoY(c_to_y_input)

	_, predicted = task_out.max(1)

	return class_index_to_string(predicted.item()), pred_concepts, image


##### https://github.com/fhvilshoj/TorchLRP/blob/74253a1be05f0be0b7c535736023408670443b6e/examples/visualization.py#L60
def heatmap(X, cmap_name="seismic"):
	cmap = plt.cm.get_cmap(cmap_name)

	if X.shape[1] in [1, 3]:  # move channel index to end + convert to np array
		X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
	if isinstance(X, torch.Tensor):  # convert tensor to np array
		X = X.detach().cpu().numpy()

	shape = X.shape
	tmp = X.sum(axis=-1) # Reduce channel axis

	tmp = project(tmp, output_range=(0, 255)).astype(int)
	tmp = cmap(tmp.flatten())[:, :3].T
	tmp = tmp.T

	shape = list(shape)
	shape[-1] = 3
	return tmp.reshape(shape).astype(np.float32)


def project(X, output_range=(0, 1)):
	absmax = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
	X /= absmax + (absmax == 0).astype(float)
	X = (X+1) / 2. # range [0, 1]
	X = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
	return X


@celery.task()
def gat_saliency(pred_concepts, input, target_id):
	filter_out = torch.zeros_like(pred_concepts)
	filter_out[:,target_id] += 1

	# Get the gradient of each input
	image_gradient = torch.autograd.grad(
		pred_concepts,
		input,
		grad_outputs=filter_out,
		retain_graph=True)[0]


	attr = heatmap(image_gradient, cmap_name='seismic')
	return attr
