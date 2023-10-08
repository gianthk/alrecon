import os.path

import yaml
from os import getlogin, path
from random import randint
import numpy as np

import dxchange
import tomopy
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
		self.dataset = ''
		self.projs = np.zeros([0, 0, 0])
		self.flats = np.zeros([0, 0])
		self.darks = np.zeros([0, 0])
		self.recon = np.zeros([0, 0, 0])
		self.theta = np.zeros(0)
		self.load_status = solara.reactive(False)
		self.loaded_file = solara.reactive(False)
		self.cor_status = solara.reactive(False)
		self.COR_range = solara.reactive((1260, 1300))
		self.COR_guess = solara.reactive(1280)

		self.proj_range_enable = solara.reactive(False)

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

	def get_phase_params(self, filename):
		try:
			self.sdd.set(dxchange.read_hdf5(filename, '/measurement/instrument/detector_motor_stack/detector_z')[0])
		except:
			print("Cannot read detector_z value")

		try:
			self.energy.set(dxchange.read_hdf5(filename, '/measurement/instrument/monochromator/energy')[0])
		except:
			print("Cannot read monochromator energy")

		try:
			camera_pixel_size = dxchange.read_hdf5(filename, '/measurement/instrument/camera/pixel_size')[0]
			magnification = dxchange.read_hdf5(filename, '/measurement/instrument/detection_system/objective/magnification')[0]
		except:
			print("Cannot read detector information (camera pixel_size; magnification)")

		self.pixelsize.set(0)
		if not magnification == 0:
			if not isnan(camera_pixel_size / magnification):
				self.pixelsize.set(camera_pixel_size / magnification)

	def load_and_normalize(self, filename):
		# global projs
		# global theta
		self.load_status.set(True)

		if self.proj_range_enable.value:
			self.projs, self.flats, self.darks, _ = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1), proj=(self.proj_range.value[0], self.proj_range.value[1], 1))
			self.theta = np.radians(dxchange.read_hdf5(filename, 'exchange/theta', slc=((self.proj_range.value[0], self.proj_range.value[1], 1),)))
		else:
			self.projs, self.flats, self.darks, self.theta = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1))

		self.loaded_file.set(True)

		self.sino_range.set([self.sino_range.value[0], self.sino_range.value[0] + self.projs.shape[1]])
		self.proj_range.set([self.proj_range.value[0], self.proj_range.value[0] + self.projs.shape[0]])

		# projs_shape.set(projs.shape)
		# flats_shape.set(flats.shape)
		# darks_shape.set(darks.shape)

		print("Dataset size: ", self.projs[:, :, :].shape[:], " - dtype: ", self.projs.dtype)
		print("Flat fields size: ", self.flats[:, :, :].shape[:])
		print("Dark fields size: ", self.darks[:, :, :].shape[:])
		print("Theta array size: ", self.theta.shape[:])

		if self.normalize_on_load.value:
			self.projs = tomopy.normalize(self.projs, self.flats, self.darks, ncore=self.ncore.value, averaging=self.averaging.value)
			print("Sinogram: normalized.")

		self.load_status.set(False)
		# COR_slice_ind.set(int(np.mean(ar.sino_range.value)))

		self.get_phase_params(filename)

		# if ar.COR_auto.value:
		# 	guess_COR()

	def guess_COR(self):
		self.cor_status.set(True)
		if self.COR_ar.algorithm.value == "Vo":
			self.COR_guess.value = tomopy.find_center_vo(self.projs, ncore=self.ncore.value)
			print("Automatic detected COR: ", self.COR_guess.value, " - tomopy.find_center_vo")
		elif self.COR_ar.algorithm.value == "TomoPy":
			self.COR_guess.value = tomopy.find_center(self.projs, self.theta)[0]
			print("Automatic detected COR: ", self.COR_guess.value, " - tomopy.find_center")

		self.COR.set(self.COR_guess.value)
		self.COR_range.set([self.COR_guess.value - 20, self.COR_guess.value + 20])
		self.cor_status.set(False)

	def myfunc(self):
		print("Hello my name is " + self.name)