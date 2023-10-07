import os.path

import yaml
from os import getlogin, path
from random import randint

import solara

def generate_title():
	titles = ["Al-Recon. CT reconstruction for dummies",
			  "Al-Recon. Have fun reconstructing",
			  "Al-Recon. Anyone can reconstruct",
			  "Al-Recon. The CT reconstruction GUI",
			  "Al-Recon. It has never been so easy",
			  "Al-Recon. CT reconstruction made simple",
			  "Al-Recon. It's business as usual"]

	return titles[randint(0, len(titles) - 1)]

def settings_file():
	# check if user settings file exists
	user_settings_file = 'alrecon/settings/' + getlogin() + '.yml'

	if path.isfile(user_settings_file):
		return user_settings_file
	else:
		return 'alrecon/settings/default.yml'

class alrecon:
	def __init__(self):
		self.algorithms = ["gridrec", "fbp_cuda_astra", "sirt_cuda_astra", "sart_cuda_astra", "cgls_cuda_astra"]
		self.title = generate_title()
		self.init_settings(settings_file())
		self.saved_info = False

	def check_settings_paths(self):
		search_key = '_dir'

		for key, val in self.settings.items():
			if search_key in key:
				if not path.isdir(val):
					print('{0}: {1} does not exist.'.format(key, val))
					self.settings[key] = '/'

	def check_path(self, var, create=False):
		if create:
			dir = path.dirname(var.value)
		else:
			dir = var.value
		if not path.isdir(dir):
			var.set('/')

	def init_settings(self, filename):
		with open(filename, "r") as file_object:
			self.settings_file = solara.reactive(os.path.basename(filename))
			self.settings = yaml.load(file_object, Loader=yaml.SafeLoader)
			self.check_settings_paths()

			# initialize app settings from YAML file
			for key, val in self.settings.items():
				exec('self.'+key + '=solara.reactive(val)')

		print('Init settings file: {0}'.format(filename))

	def load_app_settings(self, filename):
		with open(filename, "r") as file_object:
			self.settings_file.set(os.path.basename(filename))
			self.settings = yaml.load(file_object, Loader=yaml.SafeLoader)

			# some app settings
			for key, val in self.settings.items():
				exec('self.'+key + '.set(val)')

		print('Loaded settings file: {0}'.format(filename))

	def save_app_settings(self, filename):
		# update app settings dictionary to current app state
		for key, val in self.settings.items():
			exec('self.settings[\''+key+'\'] = self.'+key+'.value')

		# convert tuples to lists
		for key, val in self.settings.items():
			if type(val) is tuple:
				self.settings[key] = list(val)

		# write YAML settings file
		with open(filename, 'w') as file:
			yaml.dump(self.settings, file)

		print('Saved settings file: {0}'.format(filename))

	def myfunc(self):
		print("Hello my name is " + self.name)