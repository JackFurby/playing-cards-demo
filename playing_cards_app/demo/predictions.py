import torch
from torch.functional import F
from playing_cards_app.demo.data.utils import IndexToString, Transform
from playing_cards_app.demo.model import End2EndModel
from playing_cards_app.demo.lrp_converter import convert
import playing_cards_app.demo.lrp as lrp
from playing_cards_app.demo.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import threading


class Model():
	def __init__(
		self,
		n_img_out=1,
		use_sig=True,
		n_concepts=52,
		n_classes=6,
		concept_index_to_string="./playing_cards_app/demo/data/concepts.txt",
		concept_index_to_full_name="./playing_cards_app/demo/data/concepts_full_names.txt",
		class_index_to_string="./playing_cards_app/demo/data/classes.txt",
		class_index_to_description="./playing_cards_app/demo/data/classes_desc.txt",
		img_transform=Transform(),
		XtoCtoY_path="./playing_cards_app/demo/model_saves/XtoCtoY_converted.pth"
		):
		super(Model, self).__init__()

		self.use_sig = use_sig
		self.concept_index_to_string = IndexToString(concept_index_to_string)
		self.class_index_to_string = IndexToString(class_index_to_string)
		self.class_index_to_description = IndexToString(class_index_to_description)
		self.concept_index_to_full_name = IndexToString(concept_index_to_full_name)
		self.img_transform = img_transform

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if torch.cuda.is_available():
			print("Device:", self.device, torch.cuda.get_device_name(0))
		else:
			print("Device:", self.device)


		XtoCtoY = End2EndModel(num_classes=n_classes, num_concepts=n_concepts, num_images_per_output=n_img_out, use_sigmoid=self.use_sig, use_modulelist=False)
		XtoCtoY.load_state_dict(torch.load(XtoCtoY_path, map_location=torch.device('cpu')))
		XtoCtoY.eval()
		XtoC = convert(XtoCtoY.x_to_c_model)

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

		self.XtoC_model_1 = lrp.Sequential(*XtoC_model_1).to(self.device)
		self.XtoC_model_2 = lrp.Sequential(*XtoC_model_2).to(self.device)
		self.XtoC_model_3 = lrp.Sequential(*XtoC_model_3).to(self.device)

		self.CtoY = XtoCtoY.c_to_y_model.to(self.device)

		self.raw_input = None  # unultered input image
		self.transformed_input = None  # input image with a transformation
		self.input_name = None  # file name of input image
		self.input_path = None  # path to input image
		self.pred_concepts = None  # X to C model part output
		self.task_out = None  # C to Y model part output
		self.concepts = None  # concept out (after sigmoid or intevention)

		self.thread = threading.Semaphore()

	def predict(self, concepts=None):
		""" pass an input image through a model to get a prediction for task and concepts.
		Optionaly also generate and save concept saliency maps

		args:
			concepts (None / list of floats): List of concepts (only used for C to Y / interveneing)
		"""

		# C to Y only
		if concepts != None:
			c_to_y_input = torch.FloatTensor(concepts).unsqueeze(0).to(self.device)
		# End to end
		else:
			# Transform image and prepare for input to model
			self.transformed_input = self.img_transform(self.raw_input).to(self.device)
			self.transformed_input = self.transformed_input.unsqueeze(0)
			self.transformed_input.requires_grad_(True)
			self.transformed_input.grad = None  # Reset gradient

			# Predict concepts
			output_1 = self.XtoC_model_1.forward(self.transformed_input, explain=True, rule="alpha1beta0")
			output_2 = self.XtoC_model_2.forward(output_1, explain=True, rule="epsilon")
			self.pred_concepts = self.XtoC_model_3.forward(output_2, explain=True, rule="gradient")

			# Prepare concept predictions for task prediction
			outputs = []
			for out in [self.pred_concepts]:
				outputs.append(out)
			if self.use_sig:
				c_to_y_input = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
				sig_concepts = c_to_y_input
			else:
				c_to_y_input = torch.cat((outputs), dim=1)
				sig_concepts = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))

			concepts = sig_concepts[0].tolist()

		self.concepts = concepts

		self.task_out = self.CtoY(c_to_y_input)

		print(c_to_y_input)

	def get_results(self, sort="index"):
		""" Return model outputs and string conversion

		args:
			sort (string): how the concepts should be sorted
				options:
					index: sorted by concept index (0 first)
					softmax: sorted by softmax value (highest first)
					concept: sorted by concept value (highest first)

		returns:
			task prediction string: string name for task prediction
			task prediction description: description of task prediction
			concept_out (list of tuples): list of concept predictions in the form
				(concept ID, concept string, concept value, concept name, concept softmax score)
		"""

		_, predicted = self.task_out.max(1)

		softmax_score = F.softmax(self.pred_concepts, dim=-1)

		concept_out = []
		for idx, i in enumerate(self.concepts):
			concept_out.append((idx, self.concept_index_to_string(idx), i, self.concept_index_to_full_name(idx), softmax_score[0][idx].item()))

		indexed_concepts = [(x[0], x[1], x[2]) for x in concept_out]  # concept index, string and value in order of index

		if sort == "index":
			pass  # no need to sort
		elif sort == "softmax":
			sorted_concept_out = sorted(concept_out, key=lambda tup: tup[4], reverse=True)
		elif sort == "concept":
			sorted_concept_out = sorted(concept_out, key=lambda tup: tup[2], reverse=True)

		return self.class_index_to_string(predicted.item()), self.class_index_to_description(predicted.item()), sorted_concept_out, indexed_concepts

	def gen_saliency(self, concept_id, cmap="seismic", **kwargs):
		"""
		Generate and save saleincy map for a given concept

		args:
			concept_id (int): ID of the concept to generate a saleincy map of
			cmap (string): Name of matplotlib colour map to use
			**kwargs: additional variables which are passed directly to save_image()
		"""
		self.thread.acquire()  # add lock to avoid CUDA out of memory error
		filename = f"{concept_id}_{self.input_name}"
		# Only generate saliency map if it does not already exist
		if os.path.exists(f"{os.path.dirname(self.input_path)}/{filename}") == False:
			attr = get_saliency(self.pred_concepts, self.transformed_input, concept_id, cmap_name=cmap)
			save_image(filename=filename, img=attr.squeeze(), type="plt", **kwargs)
		self.thread.release()

		return filename

	def set_input(self, path):
		"""
		replaces the model input with a new image. Any alpha channel is removed

		args:
			image (string): file path to input image
		"""
		# remove old input + results from model and memory
		del self.raw_input
		del self.transformed_input
		del self.pred_concepts
		del self.task_out

		# add new input to class instance
		self.input_path = path
		self.input_name = os.path.basename(path)
		self.raw_input = Image.open(self.input_path).convert('RGBA')
		background = Image.new('RGBA', self.raw_input.size, (255,255,255))
		self.raw_input = Image.alpha_composite(background, self.raw_input)
		self.raw_input = self.raw_input.convert('RGB')


# https://github.com/fhvilshoj/TorchLRP/blob/74253a1be05f0be0b7c535736023408670443b6e/examples/visualization.py#L60
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
