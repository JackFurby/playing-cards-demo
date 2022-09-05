import cv2
import imageio as iio
import time

import torch
from playing_cards_app.demo.data.utils import IndexToString, Transform
from playing_cards_app.demo.model import End2EndModel
from playing_cards_app.demo.lrp_converter import convert
import playing_cards_app.demo.lrp as lrp
from PIL import Image

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


def get_camera():
	camera = iio.get_reader("<video0>")
	meta = camera.get_meta_data()
	return camera, meta


def gen_frames():
	camera, meta = get_camera()
	delay = 1/meta["fps"]
	while True:
		frame = get_frame(camera)
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		time.sleep(delay)
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def get_picture():
	time.sleep(0.3)  # make sure the camera is available (may be used by stream before)
	camera, meta = get_camera()
	return get_frame(camera)


def get_frame(camera):
	frame = camera.get_next_data()
	frame = square_crop(frame)  # discard parts of frame we cannot pass into model
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


def predict(img, concepts=None):

	# C to Y only
	if concepts != None:
		c_to_y_input = torch.FloatTensor(concepts).unsqueeze(0)

		# Get human readable outputs
		concept_out = []
		for idx, i in enumerate(concepts):
			concept_out.append((idx, concept_index_to_string(idx), i))
	# end to end
	else:
		image = Image.open(img).convert('RGBA')
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

		# Get human readable outputs
		concept_out = []
		for idx, i in enumerate(sig_concepts[0]):
			concept_out.append((idx, concept_index_to_string(idx), i.item()))

	task_out = CtoY(c_to_y_input)

	print(task_out)

	_, predicted = task_out.max(1)

	return class_index_to_string(predicted.item()), concept_out
