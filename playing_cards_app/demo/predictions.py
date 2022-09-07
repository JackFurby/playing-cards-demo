import torch
from playing_cards_app.demo.data.utils import IndexToString, Transform
from playing_cards_app.demo.model import End2EndModel
from playing_cards_app.demo.lrp_converter import convert
import playing_cards_app.demo.lrp as lrp
from playing_cards_app.demo.utils import save_image
from playing_cards_app import celery
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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


@celery.task()
def predict(image, concepts=None, gen_salience=True, input_name="", **kwargs):

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

		concepts = sig_concepts[0].tolist()

	task_out = CtoY(c_to_y_input)

	_, predicted = task_out.max(1)

	if gen_salience:
		cmap = 'seismic'
		for i in range(pred_concepts.size(1)):
			attr = get_saliency(pred_concepts, image, i, cmap_name=cmap)
			save_image(filename=f"{i}_{input_name}", img=attr.squeeze(), type="plt", **kwargs)

	return class_index_to_string(predicted.item()), concepts


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


def get_saliency(pred_concepts, input, target_id, cmap_name='seismic'):
	filter_out = torch.zeros_like(pred_concepts)
	filter_out[:,target_id] += 1

	# Get the gradient of each input
	image_gradient = torch.autograd.grad(
		pred_concepts,
		input,
		grad_outputs=filter_out,
		retain_graph=True)[0]

	attr = heatmap(image_gradient, cmap_name=cmap_name)
	return attr


def get_readable_concepts(concept_vec):
	concept_out = []
	for idx, i in enumerate(concept_vec):
		concept_out.append((idx, concept_index_to_string(idx), i))
	return concept_out
