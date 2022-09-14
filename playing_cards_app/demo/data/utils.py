import torch
import torchvision.transforms as transforms


class IndexToString(object):
	"""Convert index to the corrisponding string
	Args:
		file_path (string): path to txt file containing class ids (0 indexed) and strings
	"""

	def __init__(self, file_path):
		self.string_dict = {}
		with open(file_path) as f:
			lines = f.readlines()
			for i in lines:
				i = i.split(" ")
				key = i.pop()
				self.string_dict[int(key)] = ' '.join(i)

	def __call__(self, index):
		"""
		Args:
			index (int): index of class
		Returns:
			String: String name
		"""
		return self.string_dict[index]


class Transform():
	"""
	"""

	def __init__(self):
		self.transform = transforms.Compose([
			transforms.Resize((299, 299)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
		])

	def __call__(self, img, normalise=True):
		"""
		"""
		if normalise:
			return self.transform(img)
		else:
			return torch.as_tensor(transforms.Resize((299, 299))(img))
