import yaml
from os import getlogin, path
from pathlib import Path
import solara
from random import randint

def settings_file():
	# check if user settings file exists
	user_settings_file = 'alrecon/settings/' + getlogin() + '.yml'

	if path.isfile(user_settings_file):
		return user_settings_file
	else:
		return 'alrecon/settings/default.yml'

def load_app_settings():
	with open(settings_file(), "r") as file_object:
		alrecon_settings = yaml.load(file_object, Loader=yaml.SafeLoader)

		# locations
		master = solara.reactive(alrecon_settings['locations']['master'])
		experiment = alrecon_settings['locations']['experiment']
		recon_dir = solara.reactive(alrecon_settings['locations']['recon'])
		cor_dir = solara.reactive(alrecon_settings['locations']['cor'])
		Fiij_exe = solara.reactive(alrecon_settings['locations']['imagej']) # path to ImageJ executable

		# general settings
		ncore = solara.reactive(alrecon_settings['general']['ncore'])
		averaging = solara.reactive(alrecon_settings['normalize']['averaging'])

		# load settings
		normalize = solara.reactive(alrecon_settings['load']['normalize'])
		sino_range = solara.reactive(alrecon_settings['load']['sino_range'])
		proj_range = solara.reactive(alrecon_settings['load']['proj_range'])

		# Center Of Rotation settings
		cor = solara.reactive(alrecon_settings['cor']['cor'])
		step = solara.reactive(alrecon_settings['cor']['step'])
		autoguess = solara.reactive(alrecon_settings['cor']['autoguess'])
		cor_algorithm = solara.reactive(alrecon_settings['cor']['algorithm'])

		# phase retrieval settings
		phase_object = solara.reactive(alrecon_settings['phase-retrieval']['phase_object'])
		pixelsize = solara.reactive(alrecon_settings['phase-retrieval']['pixelsize'])
		energy = solara.reactive(alrecon_settings['phase-retrieval']['energy'])
		sdd = solara.reactive(alrecon_settings['phase-retrieval']['sdd'])
		alpha = solara.reactive(alrecon_settings['phase-retrieval']['alpha'])\

		# reconstruction settings
		algorithms = alrecon_settings['recon']['algorithms']
		algorithm = solara.reactive(alrecon_settings['recon']['algorithm'])
		num_iter = solara.reactive(alrecon_settings['recon']['num_iter'])

		# output settings
		uintconvert = solara.reactive(alrecon_settings['output']['uintconvert'])
		bitdepth = solara.reactive(alrecon_settings['output']['bitdepth'])
		circmask = solara.reactive(alrecon_settings['output']['circmask'])
		circ_mask_ratio = solara.reactive(alrecon_settings['output']['circ_mask_ratio'])

		# plotting settings
		plotreconhist = solara.reactive(alrecon_settings['plotting']['plotreconhist'])
		hist_speed = solara.reactive(alrecon_settings['plotting']['speed'])


	print('Loaded settings file: {0}'.format(settings_file()))

	return master, experiment, recon_dir, cor_dir, Fiij_exe, ncore, averaging, normalize, sino_range, proj_range, cor, step, autoguess, cor_algorithm, phase_object, pixelsize, energy, sdd, alpha, algorithms, algorithm, num_iter, uintconvert, bitdepth, circmask, circ_mask_ratio, plotreconhist, hist_speed

def generate_title():
    titles = ["Al-Recon. CT reconstruction for dummies",
              "Al-Recon. Have fun reconstructing",
              "Al-Recon. Anyone can reconstruct",
              "Al-Recon. The CT reconstruction GUI",
              "Al-Recon. It has never been so easy",
              "Al-Recon. CT reconstruction made simple",
              "Al-Recon. It's business as usual",
              ]
    return titles[randint(0, len(titles) - 1)]
