import torch
from torch import nn


class End2EndModel(torch.nn.Module):
	def __init__(self, num_classes=6, num_concepts=52, num_images_per_output=3, pretrained_weights=None, freeze=False, use_sigmoid=False, use_modulelist=True):
		super(End2EndModel, self).__init__()
		self.c_to_y_model = CtoYNet(num_concepts * num_images_per_output, num_classes)
		self.use_sigmoid = use_sigmoid
		self.use_modulelist = use_modulelist
		self.num_images_per_output = num_images_per_output

		if self.use_modulelist:
			self.x_to_c_model = PlayingCardNet(num_concepts=num_concepts, pretrained_weights=pretrained_weights, freeze=freeze)
		else:
			self.x_to_c_model = PlayingCardNet(num_classes=num_concepts, pretrained_weights=pretrained_weights, freeze=freeze)

	def forward_stage2(self, stage_1_out):
		outputs = []
		if self.use_modulelist:
			for out in stage_1_out:
				outputs.append(torch.cat(out, dim=1))
		else:
			for out in stage_1_out:
				outputs.append(out)
		if self.use_sigmoid:
			c_to_y_input = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
		else:
			c_to_y_input = torch.cat((outputs), dim=1)

		return (torch.cat(outputs, dim=1), self.c_to_y_model(c_to_y_input))

	def forward(self, x):
		if self.num_images_per_output != 1:
			# split batch into first hand card, second hand card etc.
			# Run each card batch seperatly
			split_images = []
			for image_id in range(x.size(1)):
				split_images.append(x[:, image_id, :, :, :])
		else:
			split_images = [x]  # make sure input is in same format as is there are multiple inputs

		outputs = []
		for image in split_images:
			outputs.append(self.x_to_c_model(image))
		return self.forward_stage2(outputs)


class CtoYNet(nn.Module):
	def __init__(self, num_concept_in, num_classes):
		super(CtoYNet, self).__init__()
		self.classifier = nn.Sequential(
			nn.Linear(num_concept_in, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
		return self.classifier(x)


# VGG 11 with Batch Norm
class PlayingCardNet(nn.Module):
	def __init__(self, num_classes=52, num_concepts=0, pretrained_weights=None, freeze=False):
		super(PlayingCardNet, self).__init__()
		self.num_concepts = num_concepts
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.avgPool = nn.AdaptiveAvgPool2d((7, 7))
		self.flatten = nn.Flatten()
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
		)
		# concept training requires a module list for loss
		if self.num_concepts > 0:
			# all_fc replaces fc2 from standard model
			self.all_fc = nn.ModuleList()  # separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
			for i in range(self.num_concepts):
				self.all_fc.append(nn.Linear(4096, 1))
		# basic fc for standard training
		else:
			self.out_fc = nn.Linear(4096, num_classes)

		# if weights file provided, load them
		if pretrained_weights is not None:
			self.load_state_dict(torch.load(pretrained_weights, map_location=torch.device('cpu')), strict=False)  # as we are replacing the final fc layer we can set strict to False
			print(f"Loaded weights from {pretrained_weights}")

		# if freeze == true then freeze all but fc layers
		if freeze:  # only finetune fc layer
			for name, param in self.named_parameters():
				if 'fc' not in name:
					param.requires_grad = False
			print("frozen lower model layers")

	def forward(self, x):
		x = self.features(x)
		x = self.avgPool(x)
		x = self.flatten(x)
		x = self.classifier(x)
		if self.num_concepts > 0:
			out = []
			for fc in self.all_fc:
				out.append(fc(x))
		else:
			out = self.out_fc(x)
		return out
