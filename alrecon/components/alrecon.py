import os.path

import yaml
from os import getlogin, path
from random import randint
from time import time
import numpy as np
from math import isnan

import dxchange
import tomopy
import solara

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alrecon")
logger_dxchange = logging.getLogger("dxchange")
logger_dxchange.setLevel(logging.CRITICAL)

from alrecon.components import gspreadlog

# touint should be imported from recon_utils
def touint(data_3D, dtype='uint8', range=None, quantiles=None, numexpr=True, subset=True):
    """Normalize and convert data to unsigned integer.

    Parameters
    ----------
    data_3D
        Input data.
    dtype
        Output data type ('uint8' or 'uint16').
    range : [float, float]
        Control range for data normalization.
    quantiles : [float, float]
        Define range for data normalization through input data quantiles. If range is given this input is ignored.
    numexpr : bool
        Use fast numerical expression evaluator for NumPy (memory expensive).
    subset : bool
        Use subset of the input data for quantile calculation.

    Returns
    -------
    output : uint
        Normalized data.
    """

    def convertfloat():
        return data_3D.astype(np.float32, copy=False), np.float32(data_max - data_min), np.float32(data_min)

    def convertint():
        if dtype == 'uint8':
            return convert8bit()
        elif dtype == 'uint16':
            return convert16bit()

    def convert16bit():

        data_3D_float, df, mn = convertfloat()

        if numexpr:
            import numexpr as ne

            scl = ne.evaluate('0.5+65535*(data_3D_float-mn)/df', truediv=True)
            ne.evaluate('where(scl<0,0,scl)', out=scl)
            ne.evaluate('where(scl>65535,65535,scl)', out=scl)
            return scl.astype(np.uint16)
        else:
            data_3D_float = 0.5 + 65535 * (data_3D_float - mn) / df
            data_3D_float[data_3D_float < 0] = 0
            data_3D_float[data_3D_float > 65535] = 65535
            return np.uint16(data_3D_float)

    def convert8bit():

        data_3D_float, df, mn = convertfloat()

        if numexpr:
            import numexpr as ne

            scl = ne.evaluate('0.5+255*(data_3D_float-mn)/df', truediv=True)
            ne.evaluate('where(scl<0,0,scl)', out=scl)
            ne.evaluate('where(scl>255,255,scl)', out=scl)
            return scl.astype(np.uint8)
        else:
            data_3D_float = 0.5 + 255 * (data_3D_float - mn) / df
            data_3D_float[data_3D_float < 0] = 0
            data_3D_float[data_3D_float > 255] = 255
            return np.uint8(data_3D_float)

    if range == None:

        # if quantiles is empty data is scaled based on its min and max values
        if quantiles == None:
            data_min = np.nanmin(data_3D)
            data_max = np.nanmax(data_3D)
            data_max = data_max - data_min
            return convertint()
        else:
            if subset:
                [data_min, data_max] = np.quantile(np.ravel(data_3D[0::10, 0::10, 0::10]), quantiles)
            else:
                [data_min, data_max] = np.quantile(np.ravel(data_3D), quantiles)

            return convertint()

    else:
        # ignore quantiles input if given
        if quantiles is not None:
            logger.warning('Quantiles input ignored.')

        data_min = range[0]
        data_max = range[1]
        return convertint()

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
		self.averagings = ['mean', 'median']
		self.stripe_removal_methods = ['remove_dead_stripe', 'remove_all_stripe', 'remove_large_stripe', 'remove_stripe_based_fitting']
		self.title = generate_title()
		self.init_settings(settings_file())
		self.saved_info = False
		self.worker = 'local' # 'rum'
		self.exp_time = 0.
		self.projs = np.zeros([0, 0, 0])
		self.projs_stripe = np.zeros([0, 0, 0])
		self.projs_phase = np.zeros([0, 0, 0])
		self.flats = np.zeros([0, 0])
		self.darks = np.zeros([0, 0])
		self.recon = np.zeros([0, 0, 0])
		self.theta = np.zeros(1)

		self.dataset = solara.reactive('')
		self.n_proj = solara.reactive(10001)
		self.projs_shape = solara.reactive([0, 0, 0])
		self.sino_rows = solara.reactive(2200)
		self.flats_shape = solara.reactive([0, 0, 0])
		self.darks_shape = solara.reactive([0, 0, 0])

		self.COR_range = solara.reactive((1260, 1300))
		self.COR_steps = [0.5, 1, 2, 5, 10]
		self.COR_guess = solara.reactive(1280)
		self.COR_algorithms = ["Vo", "TomoPy"]
		self.COR_slice_ind = solara.reactive(1000)  # int(projs.shape[0]/2)
		self.recon_counter = solara.reactive(0)
		self.Data_min = solara.reactive(0)
		self.Data_max = solara.reactive(0)

		self.camera_pixel_size = 0
		self.magnification = 0

		self.sino_range_enable = solara.reactive(True)
		self.proj_range_enable = solara.reactive(False)
		self.load_status = solara.reactive(False)
		self.loaded_file = solara.reactive(False)
		self.cor_status = solara.reactive(False)
		self.stripe_removal_status = solara.reactive(False)
		self.stripe_removed = solara.reactive(False)
		self.recon_status = solara.reactive(False)
		self.reconstructed = solara.reactive(False)
		self.retrieval_status = solara.reactive(False)
		self.phase_retrieved = solara.reactive(False)

		self.glog = gspreadlog.logger(key=self.gspread_key.value, experiment_name=self.experiment_name.value)

	def check_settings_paths(self):
		search_key = '_dir'

		for key, val in self.settings.items():
			if search_key in key:
				if not path.isdir(val):
					logger.warning('{0}: {1} does not exist.'.format(key, val))
					self.settings[key] = '/'

	def check_path(self, var, create=False):
		if create:
			dir = path.dirname(var.value)
			dir2 = path.dirname(dir)
		else:
			dir = var.value
			dir2 = 'nodir'

		if not (path.isdir(dir) | path.isdir(dir2)):
			var.set('/')

	def set_output_dirs(self):
		if self.auto_complete.value:
			self.recon_dir.set(self.experiment_dir.value+'scratch/'+self.experiment_name.value+'/'+self.dataset.value+'/recon')
			self.cor_dir.set(self.experiment_dir.value+'scratch/'+self.experiment_name.value+'/'+self.dataset.value+'/cor')
			self.check_path(self.recon_dir, True)
			self.check_path(self.cor_dir, True)

	def set_file_and_proj(self, dataset_path):
		self.h5file.set(dataset_path)
		self.set_n_proj(dataset_path)
		self.set_sino_rows(dataset_path)
		self.set_exp_time(dataset_path)
		self.set_phase_params(dataset_path)

		if self.proj_range_enable.value:
			if self.proj_range.value[1] > self.n_proj.value:
				self.proj_range.set([0, self.n_proj.value])
		else:
			self.proj_range.set([0, self.n_proj.value])

		if self.sino_range.value[1] > self.sino_rows.value:
			self.sino_range.set([0, self.sino_rows.value])

		self.dataset.set(path.splitext(path.basename(str(dataset_path)))[0])
		self.set_output_dirs()

	def init_settings(self, filename):
		with open(filename, "r") as file_object:
			self.settings_file = solara.reactive(os.path.basename(filename))
			self.settings = yaml.load(file_object, Loader=yaml.SafeLoader)
			self.check_settings_paths()

			# initialize app settings from YAML file
			for key, val in self.settings.items():
				exec('self.'+key + '=solara.reactive(val)')

		logger.info('Init settings file: {0}'.format(filename))

	def load_app_settings(self, filename):
		with open(filename, "r") as file_object:
			self.settings_file.set(os.path.basename(filename))
			self.settings = yaml.load(file_object, Loader=yaml.SafeLoader)

			# some app settings
			for key, val in self.settings.items():
				exec('self.'+key + '.set(val)')

		logger.info('Loaded settings file: {0}'.format(filename))

	def update_settings_dictionary(self):
		# update app settings dictionary to current app state
		for key, val in self.settings.items():
			if hasattr(self, key):
				solara_item = False
				if eval('hasattr(self.'+key+', \'value\')'):
					exec('self.settings[\'' + key + '\'] = self.' + key + '.value')
				else:
					exec('self.settings[\'' + key + '\'] = self.' + key)

		# convert tuples to lists
		for key, val in self.settings.items():
			if type(val) is tuple:
				self.settings[key] = list(val)

		# add settings values for lists (this is a patch since I don't know how to effectively log lists to gspread rows)
		self.settings['sino_start'] = self.sino_range.value[0]
		self.settings['sino_end'] = self.sino_range.value[1]
		self.settings['proj_start'] = self.proj_range.value[0]
		self.settings['proj_end'] = self.proj_range.value[1]

		# add log settings for additional items
		self.settings['worker'] = self.worker
		self.settings['exp_time'] = self.exp_time
		self.settings['n_proj'] = self.n_proj.value
		# self.settings['rot_end'] = self.projs[-1]
		self.settings['angle_start'] = np.degrees(self.theta[0])
		self.settings['angle_end'] = np.degrees(self.theta[-1])
		if self.uintconvert.value:
			self.settings['dtype'] = self.bitdepth.value
		else:
			self.settings['dtype'] = 'float32'
		self.settings['Data_min'] = self.Data_min.value
		self.settings['Data_max'] = self.Data_max.value

	def save_app_settings(self, filename):
		self.update_settings_dictionary()

		# write YAML settings file
		with open(filename, 'w') as file:
			yaml.dump(self.settings, file)

		logger.info('Saved settings file: {0}'.format(filename))

	# H5 readers
	def set_sino_rows(self, filename):
		try:
			dimension_y = int(dxchange.read_hdf5(filename, '/measurement/instrument/camera/dimension_y')[0])
			roi_size_y = int(dxchange.read_hdf5(filename, '/measurement/instrument/camera/roi/size_y')[0])
			if roi_size_y > 1:
				self.sino_rows.set(roi_size_y)
			else:
				self.sino_rows.set(dimension_y)
		except:
			logger.warning("Cannot read sinogram height.")

	def set_exp_time(self, filename):
		try:
			self.exp_time = dxchange.read_hdf5(filename, '/measurement/instrument/camera/exposure_time')[0]
		except:
			logger.warning("Cannot read exposure time.")

	def set_n_proj(self, filename):
		try:
			self.n_proj.set(int(dxchange.read_hdf5(filename, '/process/acquisition/rotation/num_angles')[0]))
		except:
			logger.warning("Cannot read n. of projections")

	def set_phase_params(self, filename):
		try:
			self.sdd.set(dxchange.read_hdf5(filename, '/measurement/instrument/detector_motor_stack/detector_z')[0])
		except:
			logger.warning("Cannot read detector_z value")

		try:
			self.energy.set(dxchange.read_hdf5(filename, '/measurement/instrument/monochromator/energy')[0])
		except:
			logger.warning("Cannot read monochromator energy")

		try:
			self.camera_pixel_size = dxchange.read_hdf5(filename, '/measurement/instrument/camera/pixel_size')[0]
			self.magnification = dxchange.read_hdf5(filename, '/measurement/instrument/detection_system/objective/magnification')[0]
		except:
			logger.warning("Cannot read detector information (camera pixel_size; magnification)")

		self.pixelsize.set(0)
		if not self.magnification == 0:
			if not isnan(self.camera_pixel_size / self.magnification):
				self.pixelsize.set(self.camera_pixel_size / self.magnification)

	def sinogram(self):

		if self.phase_object.value:
			if self.phase_retrieved.value:
				return self.projs_phase
			else:
				logger.error("Phase object was selected but phase information is not retrieved. I will continue with absorption object.")
				logger.info("Applied -log transform.")
				return tomopy.minus_log(self.projs, ncore=self.ncore.value)
		else:
			logger.info("Applied -log transform.")
			return tomopy.minus_log(self.projs, ncore=self.ncore.value)

	def load_and_normalize(self, filename):
		self.load_status.set(True)

		if not self.proj_range_enable.value:
			self.proj_range.set([0, self.n_proj.value])

		if not self.sino_range_enable.value:
			self.sino_range.set([0, self.sino_rows.value])

		self.projs, self.flats, self.darks, _ = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1), proj=(self.proj_range.value[0], self.proj_range.value[1], 1))
		self.theta = np.radians(dxchange.read_hdf5(filename, 'exchange/theta', slc=((self.proj_range.value[0], self.proj_range.value[1], 1),)))

		# if self.proj_range_enable.value:
		# 	self.projs, self.flats, self.darks, _ = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1), proj=(self.proj_range.value[0], self.proj_range.value[1], 1))
		# 	self.theta = np.radians(dxchange.read_hdf5(filename, 'exchange/theta', slc=((self.proj_range.value[0], self.proj_range.value[1], 1),)))
		# else:
		# 	self.projs, self.flats, self.darks, self.theta = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1))

		self.loaded_file.set(True)
		# self.dataset.set(path.splitext(path.basename(filename))[0])

		self.sino_range.set([self.sino_range.value[0], self.sino_range.value[0] + self.projs.shape[1]])
		self.proj_range.set([self.proj_range.value[0], self.proj_range.value[0] + self.projs.shape[0]])

		self.projs_shape.set(self.projs.shape)
		self.flats_shape.set(self.flats.shape)
		self.darks_shape.set(self.darks.shape)

		# logger.info("Dataset size: ", self.projs[:, :, :].shape[:], " - dtype: ", self.projs.dtype)
		logger.info("Dataset size: {0} x {1} x {2} - dtype: {3}".format(*self.projs[:, :, :].shape[:], self.projs.dtype))
		logger.info("Flat fields size: {0} x {1} x {2}".format(*self.flats[:, :, :].shape[:]))
		logger.info("Dark fields size: {0} x {1} x {2}".format(*self.darks[:, :, :].shape[:]))
		logger.info("Theta array size: {0}".format(*self.theta.shape[:]))

		if self.normalize_on_load.value:
			self.projs = tomopy.normalize(self.projs, self.flats, self.darks, ncore=self.ncore.value, averaging=self.averaging.value)
			logger.info("Sinogram: normalized.")

		self.load_status.set(False)
		self.COR_slice_ind.set(int(np.mean(self.sino_range.value)))

		# self.set_phase_params(filename)

		if self.COR_auto.value:
			self.guess_COR()

	def guess_COR(self):
		self.cor_status.set(True)
		if self.COR_algorithm.value == "Vo":
			self.COR_guess.value = tomopy.find_center_vo(self.projs, ncore=self.ncore.value)
			logger.info("Automatic detected COR: {0} - tomopy.find_center_vo".format(self.COR_guess.value))
		elif self.COR_algorithm.value == "TomoPy":
			self.COR_guess.value = tomopy.find_center(self.projs, self.theta)[0]
			logger.info("Automatic detected COR: {0} - tomopy.find_center".format(self.COR_guess.value))

		self.COR.set(self.COR_guess.value)
		self.COR_range.set([self.COR_guess.value - 20, self.COR_guess.value + 20])
		self.cor_status.set(False)

	def write_cor(self):
		self.cor_status.set(True)
		tomopy.write_center(self.projs,
		                    self.theta,
		                    self.cor_dir.value,
		                    [self.COR_range.value[0], self.COR_range.value[1], self.COR_step.value],
		                    ind=int(self.COR_slice_ind.value - self.sino_range.value[0])
		                    )
		logger.info("Reconstructed slice with COR range: {0} - {1}, step: {2}".format(self.COR_range.value[0], self.COR_range.value[1], self.COR_step.value))
		self.cor_status.set(False)

	def reconstruct_dataset(self):
		self.recon_status.set(True)

		if 'cuda_astra' in self.algorithm.value:
			if 'fbp' in self.algorithm.value:
				options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
			elif 'sirt' in self.algorithm.value:
				options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': self.num_iter.value}
			elif 'sart' in self.algorithm.value:
				options = {'proj_type': 'cuda', 'method': 'SART_CUDA', 'num_iter': self.num_iter.value}
			elif 'cgls' in self.algorithm.value:
				options = {'proj_type': 'cuda', 'method': 'CGLS_CUDA', 'num_iter': self.num_iter.value}
			else:
				logger.warning("Algorithm option not recognized. Will reconstruct with ASTRA FBP CUDA.")
				options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
			self.recon = tomopy.recon(self.sinogram(), self.theta, center=self.COR.value, algorithm=tomopy.astra, options=options, ncore=1)
		else:
			self.recon = tomopy.recon(self.sinogram(), self.theta, center=self.COR.value, algorithm=self.algorithm.value, sinogram_order=False, ncore=self.ncore.value)

		if self.phase_object.value:
			if not self.phase_retrieved.value:
				solara.Error("Phase info not retrieved! I will reconstruct an absorption dataset.", text=False, dense=True, outlined=False)

		logger.info("Dataset reconstructed.")
		self.recon_status.set(False)
		self.reconstructed.set(True)
		self.recon_counter.set(self.recon_counter.value + 1)

		if (self.Data_min.value == 0) & (self.Data_max.value == 0):
			recon_subset = self.recon[0::10, 0::10, 0::10]

			# Estimate GV range from data histogram (0.01 and 0.99 quantiles)
			[range_min, range_max] = np.quantile(recon_subset.ravel(), [0.01, 0.99])
			logger.info("1% quantile: {0}".format(range_min))
			logger.info("99% quantile: {0}".format(range_max))
			self.Data_min.set(round(range_min, 5))
			self.Data_max.set(round(range_max, 5))

	def write_recon(self):
		fileout = self.recon_dir.value + '/slice.tiff'
		self.recon_status.set(True)

		if self.uintconvert.value:
			if self.circmask.value:
				dxchange.writer.write_tiff_stack(
					tomopy.circ_mask(touint(self.recon, self.bitdepth.value, [self.Data_min.value, self.Data_max.value]),
					                 axis=0, ratio=self.circmask_ratio.value),
					fname=fileout, dtype=self.bitdepth.value, axis=0, digit=4, start=0, overwrite=True)
			else:
				dxchange.writer.write_tiff_stack(
					touint(self.recon, self.bitdepth.value, [self.Data_min.value, self.Data_max.value]),
					fname=fileout, dtype=self.bitdepth.value, axis=0, digit=4, start=0, overwrite=True)
		else:
			if self.circmask.value:
				dxchange.writer.write_tiff_stack(tomopy.circ_mask(self.recon, axis=0, ratio=self.circmask_ratio.value),
				                                 fname=fileout, axis=0, digit=4, start=0, overwrite=True)
			else:
				dxchange.writer.write_tiff_stack(self.recon, fname=fileout, axis=0, digit=4, start=0, overwrite=True)
		self.recon_status.set(False)
		logger.info("Dataset written to disk.")

	def remove_stripe(self):
		if self.stripe_removal_method.value == 'remove_dead_stripe':
			self.stripe_removal_status.set(True)
			self.projs_stripe = tomopy.prep.stripe.remove_dead_stripe(self.projs, snr=self.stripe_removal_snr.value, size=self.stripe_removal_size.value, ncore=self.ncore.value)
			logger.info("Stripes removed with method: {}\n".format(str(self.stripe_removal_method)))
			self.stripe_removal_status.set(False)
			self.stripe_removed.set(True)
		else:
			logger.warning("Stripe removal method not implemented.")

	def retrieve_phase(self):
		self.retrieval_status.set(True)
		phase_start_time = time()
		self.projs_phase = tomopy.retrieve_phase(self.projs, pixel_size=0.0001 * self.pixelsize.value, dist=0.1 * self.sdd.value, energy=self.energy.value, alpha=self.alpha.value, pad=self.pad.value, ncore=self.ncore.value, nchunk=None)
		phase_end_time = time()
		phase_time = phase_end_time - phase_start_time
		logger.info("Phase retrieval time: {} s\n".format(str(phase_time)))
		self.retrieval_status.set(False)
		self.phase_retrieved.set(True)

	def cluster_run(self):
		logger.info('Logging recon to master...')
		self.worker = 'rum'
		if not self.proj_range_enable.value:
			self.proj_range.set([0, self.n_proj.value])

		if not self.sino_range_enable.value:
			self.sino_range.set([0, self.sino_rows.value])

		self.update_settings_dictionary()
		self.glog.log_to_gspread(self.settings)

		# logger.info('launch recon on rum...')

	# del variables
