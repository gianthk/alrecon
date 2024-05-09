"""
This is the main class of Al-recon: a Computed Tomography reconstruction web application running on solara.
For more information, visit the project homepage:
    https://github.com/gianthk/alrecon
"""

__author__ = ["Gianluca Iori"]
__date_created__ = "2023-08-01"
__date__ = "2024-03-19"
__copyright__ = "Copyright (c) 2024, SESAME"
__docformat__ = "restructuredtext en"
__license__ = "MIT"
__maintainer__ = "Gianluca Iori"
__email__ = "gianthk.iori@gmail.com"

# import os.path

# import yaml


from os import path
import os
import subprocess
from random import randint
from time import time
import numpy as np
from math import isnan
import pandas as pd

import dxchange
import tomopy
import solara

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alrecon")
logger_dxchange = logging.getLogger("dxchange")
logger_slurm = logging.getLogger("slurm")
logger_dxchange.setLevel(logging.CRITICAL)

from alrecon.components import gspreadlog, slurm
from alrecon.components.recon_utils import touint
from alrecon.components.general_tools import settings_file, load_app_settings, save_app_settings, init_settings


def generate_title():
    titles = [
        "Al-Recon. CT reconstruction for dummies",
        "Al-Recon. Have fun reconstructing",
        "Al-Recon. Anyone can reconstruct",
        "Al-Recon. The CT reconstruction GUI",
        "Al-Recon. It has never been so easy",
        "Al-Recon. CT reconstruction made simple",
        "Al-Recon. It's business as usual",
    ]

    titles = ["Al-Recon. It has never been so easy"]

    return titles[randint(0, len(titles) - 1)]


class alrecon:
    def __init__(self):
        self.algorithms = ["gridrec", "fbp_cuda_astra", "sirt_cuda_astra", "sart_cuda_astra", "cgls_cuda_astra"]
        self.averagings = ["mean", "median"]
        self.stripe_removal_methods = ["remove_dead_stripe", "remove_large_stripe", "remove_stripe_based_sorting", "remove_all_stripe"]
        self.nodes = ["rum"]
        self.title = generate_title()
        self.init_settings(settings_file())
        self.saved_info = False
        self.worker = "local"  # 'rum'
        self.exp_time = 0.0
        self.theta = np.zeros(1)
        self.init_3Darrays()
        self.proj0 = np.zeros([0, 0])

        self.dataset = solara.reactive("")
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
        self.loaded_file = solara.reactive(False)
        self.cor_status = solara.reactive(False)
        self.stripe_removal_status = solara.reactive(False)
        self.stripe_removed = solara.reactive(False)
        self.reconstructed = solara.reactive(False)
        self.hist_count = solara.reactive(0)
        self.normalized = solara.reactive(False)
        self.stitched = solara.reactive(False)
        self.phase_retrieved = solara.reactive(False)

        self.load_status = solara.reactive(False)
        self.recon_status = solara.reactive(False)
        self.write_status = solara.reactive(False)
        self.retrieval_status = solara.reactive(False)
        self.stitching_status = solara.reactive(False)

        self.master = pd.DataFrame()
        #       self.master = pd.DataFrame({'dataset': ['bee_yazeed-20231001T170032', 'fiber_test_fast-20230731T185659'], 'COR': [1280, 2000], 'recon_dir': ['/tmp/Yazeed/wasp/bee_yazeed-20231001T170032/recon_phase_alpha0.0002',
        # '/home/gianthk/Data/BEATS/IH/scratch/pippo/recon']})
        self.attempt_glog_init()

    def attempt_glog_init(self):
        try:
            self.glog = gspreadlog.logger(key=self.gspread_key.value, experiment_name=self.experiment_name.value)
        except:
            logger.error(
                "Could not initialize gspread logger. Make sure that the gspread_key specified in your settings file exists, and that you have correct permissions."
            )
            self.glog = None
            self.gspread_logging.set(False)

    def read_gspread_master(self):
        # attempt gspread logger initialization if gspread_logging is selected but the logger is not initialized yet
        if self.gspread_logging.value:
            if self.glog is None:
                self.attempt_glog_init()

        # read master Google spreadsheet with reconstruction information
        if self.gspread_logging.value:
            self.master, _ = self.glog.read_gspread_master(self.settings)
            logger.info("Gspread master file read...")

    def kill_all_imagej_processes(self):
        command = "for pid in $(ps -ef | grep '{0}' | awk ".format(self.imagej_launcher.value) + "'{print $2}'); do kill -9 $pid; done"
        subprocess.run(command, shell=True)

    def init_3Darrays(self):
        self.projs = np.zeros([0, 0, 0])
        self.projs_stripe = np.zeros([0, 0, 0])
        self.projs_phase = np.zeros([0, 0, 0])
        self.recon = np.zeros([0, 0, 0])

        if not self.separate_flats.value:
            self.flats = np.zeros([0, 0, 0])

        if not self.separate_darks.value:
            self.darks = np.zeros([0, 0, 0])

    def check_settings_paths(self):
        search_key = "_dir"

        for key, val in self.settings.items():
            if search_key in key:
                if not path.isdir(val):
                    logger.warning("{0}: {1} does not exist.".format(key, val))
                    self.settings[key] = "/"

    def check_path(self, var, create=False):
        if create:
            dir = path.dirname(var.value)
            dir2 = path.dirname(dir)
        else:
            dir = var.value
            dir2 = "nodir"

        if not (path.isdir(dir) | path.isdir(dir2)):
            var.set("/")

    def __set_output_dirs(self):
        if self.auto_complete.value:
            self.recon_dir.set(self.experiment_dir.value + "scratch/" + self.experiment_name.value + "/" + self.dataset.value + "/recon")
            self.cor_dir.set(self.experiment_dir.value + "scratch/" + self.experiment_name.value + "/" + self.dataset.value + "/cor")
            self.check_path(self.recon_dir, True)
            self.check_path(self.cor_dir, True)

    def _set_output_dirs(self):
        if self.auto_complete.value:
            self.recon_dir.set(path.join(f"{self.experiment_dir.value}_recon", self.experiment_name.value, self.dataset.value, "recon"))
            self.cor_dir.set(path.join(f"{self.experiment_dir.value}_recon", self.experiment_name.value, self.dataset.value, "cor"))
            self.check_path(self.recon_dir, True)
            self.check_path(self.cor_dir, True)

    def define_recon_dir_BEATS(self, number_heading_directories=5):
        ### first 4 elements are defining the root at BEATS, + '/' what gets stripped to ''
        ### but since we know the path for BEATS, we could also take a hardcoded thing
        path_ = os.path.abspath(self.experiment_dir.value)
        elements_path = path_.split(os.sep)[:number_heading_directories]
        elements_path[0] = "/"
        dir_recon_root = os.path.join(*elements_path)
        dir_recon_root = f"{dir_recon_root}_recon"
        return dir_recon_root

    def set_output_dirs(self):
        if self.auto_complete.value:
            if self.inhouse.value:
                # CHECK THESE PATHS
                self.recon_dir.set(path.join(self.experiment_dir.value, "scratch", self.experiment_name.value, self.dataset.value, "recon"))
                self.cor_dir.set(path.join(self.experiment_dir.value, "scratch", self.experiment_name.value, self.dataset.value, "cor"))
            else:
                # out_path = path.join(f'{self.experiment_dir.value}_recon', self.experiment_name.value, self.dataset.value,  'recon')
                self.dir_recon_root = self.define_recon_dir_BEATS()
                self.recon_dir.set(path.join(self.dir_recon_root, self.experiment_name.value, self.dataset.value, "recon"))
                self.cor_dir.set(path.join(self.dir_recon_root, self.experiment_name.value, self.dataset.value, "cor"))

            self.check_path(self.recon_dir, True)
            self.check_path(self.cor_dir, True)

    def update_recon_dir(self):
        # Update reconstruction directory with '_phase' string.
        # USed when phase retrieval switch is toggled

        if self.phase_object.value:
            if "_phase" not in self.recon_dir.value:
                self.recon_dir.set(self.recon_dir.value + "_phase")
        else:
            if "_phase" in self.recon_dir.value:
                self.recon_dir.set(self.recon_dir.value[:-6])

    def set_flats_file(self, dataset_path):
        self.h5file_flats.set(dataset_path)

    def set_darks_file(self, dataset_path):
        self.h5file_darks.set(dataset_path)

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
        self.set_proj0()

        self.loaded_file.set(False)
        self.stitched.set(False)
        self.stripe_removed.set(False)
        self.normalized.set(False)
        self.phase_retrieved.set(False)
        self.reconstructed.set(False)

    def init_settings(self, filename):
        init_settings(self, filename)
        logger.info("Init settings file: {0}".format(filename))

    def load_app_settings(self, filename):
        load_app_settings(self, filename)
        logger.info("Loaded settings file: {0}".format(filename))

    def update_settings_dictionary(self):
        # update app settings dictionary to current app state
        for key, val in self.settings.items():
            if hasattr(self, key):
                solara_item = False
                if eval("hasattr(self." + key + ", 'value')"):
                    exec("self.settings['" + key + "'] = self." + key + ".value")
                else:
                    exec("self.settings['" + key + "'] = self." + key)

        # convert tuples to lists
        for key, val in self.settings.items():
            if type(val) is tuple:
                self.settings[key] = list(val)

        # add settings values for lists (this is a patch since I don't know how to effectively log lists to gspread rows)
        self.settings["sino_start"] = self.sino_range.value[0]
        self.settings["sino_end"] = self.sino_range.value[1]
        self.settings["proj_start"] = self.proj_range.value[0]
        self.settings["proj_end"] = self.proj_range.value[1]

        # add log settings for additional items
        self.settings["worker"] = self.worker
        self.settings["exp_time"] = self.exp_time
        self.settings["n_proj"] = self.n_proj.value
        # self.settings['rot_end'] = self.projs[-1]
        self.settings["angle_start"] = np.degrees(self.theta[0])
        self.settings["angle_end"] = np.degrees(self.theta[-1])
        if self.uintconvert.value:
            self.settings["dtype"] = self.bitdepth.value
        else:
            self.settings["dtype"] = "float32"
        self.settings["Data_min"] = self.Data_min.value
        self.settings["Data_max"] = self.Data_max.value

    def save_app_settings(self, filename):
        save_app_settings(self, filename)
        logger.info("Saved settings file: {0}".format(filename))

    # H5 readers
    def set_sino_rows(self, filename):
        try:
            dimension_y = int(dxchange.read_hdf5(filename, "/measurement/instrument/camera/dimension_y")[0])
            roi_size_y = int(dxchange.read_hdf5(filename, "/measurement/instrument/camera/roi/size_y")[0])
            if roi_size_y > 1:
                self.sino_rows.set(roi_size_y)
            else:
                self.sino_rows.set(dimension_y)
        except:
            logger.warning("Cannot read sinogram height.")

    def set_exp_time(self, filename):
        try:
            self.exp_time = dxchange.read_hdf5(filename, "/measurement/instrument/camera/exposure_time")[0]
        except:
            logger.warning("Cannot read exposure time.")

    def set_n_proj(self, filename):
        try:
            self.n_proj.set(int(dxchange.read_hdf5(filename, "/process/acquisition/rotation/num_angles")[0]))
        except:
            logger.warning("Cannot read n. of projections")

    def set_phase_params(self, filename):
        try:
            self.sdd.set(np.round(dxchange.read_hdf5(filename, "/measurement/instrument/detector_motor_stack/detector_z")[0]))
        except:
            logger.warning("Cannot read detector_z value")

        try:
            self.energy.set(dxchange.read_hdf5(filename, "/measurement/instrument/monochromator/energy")[0])
        except:
            logger.warning("Cannot read monochromator energy")

        try:
            self.camera_pixel_size = dxchange.read_hdf5(filename, "/measurement/instrument/camera/pixel_size")[0]
            float(self.camera_pixel_size)
        except:
            logger.warning("Cannot read camera pixel_size")
            self.camera_pixel_size = 0

        if self.camera_pixel_size == 0:
            if "PCO" in str(dxchange.read_hdf5(filename, "/measurement/instrument/camera/manufacturer")[0]):
                self.camera_pixel_size = 6.5
            elif "FLIR" in str(dxchange.read_hdf5(filename, "/measurement/instrument/camera/manufacturer")[0]):
                self.camera_pixel_size = 4.5

        try:
            detector = dxchange.read_hdf5(filename, "/measurement/instrument/detection_system/detector_name")[0]
            if detector == 1:
                lens_sel_x = dxchange.read_hdf5(filename, "/measurement/instrument/detector_motor_stack/detector1_setup/lens_sel_x")[0]
                if lens_sel_x < 0:
                    self.magnification = dxchange.read_hdf5(filename, "/measurement/instrument/detector_motor_stack/detector1_setup/magnification1")[0]
                elif lens_sel_x > 0:
                    self.magnification = dxchange.read_hdf5(filename, "/measurement/instrument/detector_motor_stack/detector1_setup/magnification2")[0]
            elif detector == 2:
                self.magnification = dxchange.read_hdf5(filename, "/measurement/instrument/detector_motor_stack/detector2_setup/magnification")[0]
            elif detector == 3:
                self.magnification = dxchange.read_hdf5(filename, "/measurement/instrument/detector_motor_stack/detector3_setup/magnification")[0]
        except:
            logger.warning("Cannot read detector information (detector type; magnification)")

        self.pixelsize.set(0)
        if not self.magnification == 0:
            if not isnan(self.camera_pixel_size / self.magnification):
                self.pixelsize.set(self.camera_pixel_size / self.magnification)

    def sinogram(self):
        # Return sinogram for pre-processing
        if (self.phase_object.value is True) & (self.phase_retrieved.value is True):
            return self.projs_phase
        else:
            if self.stripe_remove.value:
                if self.stripe_removed.value:
                    return self.projs_stripe
                else:
                    logger.error("Stripe removal was selected but the sinogram has not been processed. I will proceed without stripe removal.")
                    return self.projs
            else:
                return self.projs

    def processed_sinogram(self):
        # Return processed sinogram for reconstruction job

        # normalize if not yet normalized
        if self.normalized.value is False:
            self.normalize_sinogram()

        # stitch if extended FOV is selected but sinogram was not stitched
        if (self.extended_FOV.value is True) & (self.stitched.value is False):
            self.sino_360_to_180()

        # phase retrieval
        if (self.phase_object.value is True) & (self.phase_retrieved.value is False):
            self.retrieve_phase()

        # log transform
        logger.info("Sinogram: applied -log transform.")
        return tomopy.minus_log(self.sinogram(), ncore=self.ncore.value)

    def normalize_sinogram(self):
        self.projs = tomopy.normalize(self.projs, self.flats, self.darks, ncore=self.ncore.value, averaging=self.averaging.value)
        self.normalized.set(True)
        logger.info("Sinogram: normalized.")

    def set_proj0(self):
        tmp, _, _, _ = dxchange.read_sesame_beats(self.h5file.value, exchange_rank=0, proj=(0, 1, 1))
        self.proj0 = tmp[0, :, :]

    def load_and_normalize(self, filename, filename_flats="", filename_darks=""):
        # free some space
        del self.projs, self.projs_phase, self.projs_stripe, self.recon
        if not self.separate_flats.value:
            del self.flats

        if not self.separate_darks.value:
            del self.darks

        # reinitialize 3D arrays
        self.init_3Darrays()

        self.load_status.set(True)

        if not self.proj_range_enable.value:
            self.proj_range.set([0, self.n_proj.value])

        if not self.sino_range_enable.value:
            self.sino_range.set([0, self.sino_rows.value])

        self.projs, self.flats, self.darks, self.theta = dxchange.read_sesame_beats(
            filename,
            exchange_rank=0,
            sino=(self.sino_range.value[0], self.sino_range.value[1], 1),
            proj=(self.proj_range.value[0], self.proj_range.value[1], 1),
        )

        if (self.separate_flats.value) & (filename_flats != ""):
            _, self.flats, _, _ = dxchange.read_sesame_beats(filename_flats, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1))

        if (self.separate_darks.value) & (filename_darks != ""):
            _, _, self.darks, _ = dxchange.read_sesame_beats(filename_darks, exchange_rank=0, sino=(self.sino_range.value[0], self.sino_range.value[1], 1))

        self.loaded_file.set(True)

        self.sino_range.set([self.sino_range.value[0], self.sino_range.value[0] + self.projs.shape[1]])
        self.proj_range.set([self.proj_range.value[0], self.proj_range.value[0] + self.projs.shape[0]])

        self.projs_shape.set(self.projs.shape)
        self.flats_shape.set(self.flats.shape)
        self.darks_shape.set(self.darks.shape)

        logger.info("Dataset size: {0} x {1} x {2} - dtype: {3}".format(*self.projs[:, :, :].shape[:], self.projs.dtype))
        logger.info("Flat fields size: {0} x {1} x {2}".format(*self.flats[:, :, :].shape[:]))
        logger.info("Dark fields size: {0} x {1} x {2}".format(*self.darks[:, :, :].shape[:]))
        logger.info("Theta array size: {0}".format(*self.theta.shape[:]))

        if self.normalize_on_load.value:
            self.normalize_sinogram()

        self.load_status.set(False)
        self.COR_slice_ind.set(int(np.mean(self.sino_range.value)))

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

    def write_overlap(self):
        self.cor_status.set(True)
        for overlap in range(self.COR_range.value[0], self.COR_range.value[1], self.COR_step.value):
            fileout = f"{self.cor_dir.value}" + f"/{overlap}.tiff"
            projs_stitch = tomopy.sino_360_to_180(self.projs[:, 0:1, :], overlap=overlap, rotation=self.overlap_side.value)
            projs_stitch = tomopy.minus_log(projs_stitch, ncore=self.ncore.value)
            theta = tomopy.angles(projs_stitch.shape[0])
            COR = projs_stitch.shape[2] / 2
            recon = tomopy.recon(projs_stitch, theta, center=COR, algorithm="gridrec", ncore=self.ncore.value)
            dxchange.writer.write_tiff_stack(recon, fname=fileout, axis=0, digit=4, start=0, overwrite=True)
        self.cor_status.set(False)

    def write_cor(self):
        self.cor_status.set(True)
        tomopy.write_center(
            self.projs,
            self.theta,
            self.cor_dir.value,
            [self.COR_range.value[0], self.COR_range.value[1], self.COR_step.value],
            ind=int(self.COR_slice_ind.value - self.sino_range.value[0]),
        )
        logger.info("Reconstructed slice with COR range: {0} - {1}, step: {2}".format(self.COR_range.value[0], self.COR_range.value[1], self.COR_step.value))
        self.cor_status.set(False)

    def reconstruct_dataset(self):
        self.recon_status.set(True)

        if "cuda_astra" in self.algorithm.value:
            if "fbp" in self.algorithm.value:
                options = {"proj_type": "cuda", "method": "FBP_CUDA"}
            elif "sirt" in self.algorithm.value:
                options = {"proj_type": "cuda", "method": "SIRT_CUDA", "num_iter": self.num_iter.value}
            elif "sart" in self.algorithm.value:
                options = {"proj_type": "cuda", "method": "SART_CUDA", "num_iter": self.num_iter.value}
            elif "cgls" in self.algorithm.value:
                options = {"proj_type": "cuda", "method": "CGLS_CUDA", "num_iter": self.num_iter.value}
            else:
                logger.warning("Algorithm option not recognized. Will reconstruct with ASTRA FBP CUDA.")
                options = {"proj_type": "cuda", "method": "FBP_CUDA"}
            self.recon = tomopy.recon(self.processed_sinogram(), self.theta, center=self.COR.value, algorithm=tomopy.astra, options=options, ncore=1)
        else:
            self.recon = tomopy.recon(
                self.processed_sinogram(), self.theta, center=self.COR.value, algorithm=self.algorithm.value, sinogram_order=False, ncore=self.ncore.value
            )

        if self.phase_object.value:
            if not self.phase_retrieved.value:
                solara.Error("Phase info not retrieved! I will reconstruct an absorption dataset.", text=False, dense=True, outlined=False)

        logger.info("Dataset reconstructed.")
        self.recon_status.set(False)
        self.reconstructed.set(True)
        self.recon_counter.set(self.recon_counter.value + 1)
        self.hist_count.set(self.hist_count.value + 1)

        if (self.Data_min.value == 0) & (self.Data_max.value == 0):
            recon_subset = self.recon[0::10, 0::10, 0::10]

            # Estimate GV range from data histogram (0.1 % and 99.9 % quantiles)
            # need to add a circular mask before hist
            [range_min, range_max] = np.quantile(recon_subset.ravel(), [0.0001, 0.9999])
            logger.info("0.01% quantile: {0}".format(range_min))
            logger.info("99.99% quantile: {0}".format(range_max))
            self.Data_min.set(round(range_min, 5))
            self.Data_max.set(round(range_max, 5))

    def write_recon(self):
        fileout = self.recon_dir.value + "/slice.tiff"
        self.write_status.set(True)

        if self.uintconvert.value:
            if self.circmask.value:
                dxchange.writer.write_tiff_stack(
                    tomopy.circ_mask(
                        touint(self.recon, self.bitdepth.value, [self.Data_min.value, self.Data_max.value]), axis=0, ratio=self.circmask_ratio.value
                    ),
                    fname=fileout,
                    dtype=self.bitdepth.value,
                    axis=0,
                    digit=4,
                    start=0,
                    overwrite=True,
                )
            else:
                dxchange.writer.write_tiff_stack(
                    touint(self.recon, self.bitdepth.value, [self.Data_min.value, self.Data_max.value]),
                    fname=fileout,
                    dtype=self.bitdepth.value,
                    axis=0,
                    digit=4,
                    start=0,
                    overwrite=True,
                )
        else:
            if self.circmask.value:
                dxchange.writer.write_tiff_stack(
                    tomopy.circ_mask(self.recon, axis=0, ratio=self.circmask_ratio.value), fname=fileout, axis=0, digit=4, start=0, overwrite=True
                )
            else:
                dxchange.writer.write_tiff_stack(self.recon, fname=fileout, axis=0, digit=4, start=0, overwrite=True)

        os.chmod(path.dirname(fileout), 0o0777)
        self.write_status.set(False)
        logger.info("Dataset written to disk.")

    def remove_stripes(self):
        # implemented stripe removal methods: 'remove_dead_stripe', 'remove_large_stripe', 'remove_stripe_based_sorting', 'remove_all_stripe'

        self.stripe_removal_status.set(True)
        if self.stripe_removal_method.value == "remove_dead_stripe":
            self.projs_stripe = tomopy.prep.stripe.remove_dead_stripe(self.projs, snr=self.snr.value, size=self.size.value, ncore=self.ncore.value)
        elif self.stripe_removal_method.value == "remove_large_stripe":
            self.projs_stripe = tomopy.prep.stripe.remove_large_stripe(
                self.projs, snr=self.snr.value, size=self.size.value, drop_ratio=self.drop_ratio.value, norm=self.norm.value, ncore=self.ncore.value
            )
        elif self.stripe_removal_method.value == "remove_stripe_based_sorting":
            self.projs_stripe = tomopy.prep.stripe.remove_stripe_based_sorting(self.projs, size=self.size.value, dim=self.dim.value, ncore=self.ncore.value)
        elif self.stripe_removal_method.value == "remove_all_stripe":
            self.projs_stripe = tomopy.prep.stripe.remove_all_stripe(
                self.projs, snr=self.snr.value, la_size=self.la_size.value, sm_size=self.sm_size.value, dim=self.dim.value, ncore=self.ncore.value
            )
        else:
            logger.error("Stripe removal method not implemented.")

        logger.info("Stripes removed with method: {}\n".format(str(self.stripe_removal_method.value)))
        self.stripe_removal_status.set(False)
        self.stripe_removed.set(True)

    def sino_360_to_180(self):
        if self.normalized.value is False:
            self.normalize_sinogram()

        self.stitching_status.set(True)
        self.projs = tomopy.sino_360_to_180(self.projs, overlap=self.overlap.value, rotation=self.overlap_side.value)
        self.theta = tomopy.angles(self.projs.shape[0])
        self.COR.set(self.projs.shape[2] / 2)
        self.projs_shape.set(self.projs.shape)
        self.stitching_status.set(False)
        self.stitched.set(True)
        logger.info("Sinogram: stitched to 180 degrees.")

    def retrieve_phase(self):
        if self.normalized.value is False:
            self.normalize_sinogram()

        self.retrieval_status.set(True)
        phase_start_time = time()

        if (self.stripe_remove.value is True) & (self.stripe_removed.value is True):
            self.projs_phase = tomopy.retrieve_phase(
                self.projs_stripe,
                pixel_size=0.0001 * self.pixelsize.value,
                dist=0.1 * self.sdd.value,
                energy=self.energy.value,
                alpha=self.alpha.value,
                pad=self.pad.value,
                ncore=self.ncore.value,
                nchunk=None,
            )
        else:
            self.projs_phase = tomopy.retrieve_phase(
                self.projs,
                pixel_size=0.0001 * self.pixelsize.value,
                dist=0.1 * self.sdd.value,
                energy=self.energy.value,
                alpha=self.alpha.value,
                pad=self.pad.value,
                ncore=self.ncore.value,
                nchunk=None,
            )
        phase_end_time = time()
        phase_time = phase_end_time - phase_start_time
        logger.info("Sinogram: phase retrieved in time: {} s\n".format(str(phase_time)))
        self.retrieval_status.set(False)
        self.phase_retrieved.set(True)

    def cluster_run(self, worker="rum"):
        """Logs reconstruction info to master google spreadsheet using gspread. Generates and submit a slurm reconstruction job file to HPC cluster."""

        # apply sino_range setting
        if not self.recon_sino_range.value:
            self.sino_range_enable.set(False)

        # apply proj_range setting
        if not self.recon_proj_range.value:
            self.proj_range_enable.set(False)

        # attempt gspread logger initialization if gspread_logging is selected but the logger is not initialized yet
        if self.gspread_logging.value:
            if self.glog is None:
                self.attempt_glog_init()

        # log reconstruction information to master Google spreadsheet
        if self.gspread_logging.value:
            logger.info("Logging recon to master...")
            self.worker = worker
            if not self.proj_range_enable.value:
                self.proj_range.set([0, self.n_proj.value])

            if not self.sino_range_enable.value:
                self.sino_range.set([0, self.sino_rows.value])

            self.update_settings_dictionary()
            self.glog.log_to_gspread(self.settings)

        # write Slurm reconstruction job
        # logger.info('Writing Slurm reconstruction job.')

        # initialize slurm job instance with
        job = slurm.slurmjob(job_name=self.dataset.value, job_dir=str(path.dirname(self.recon_dir.value)))

        job.write_header(alrecon_state=self)
        job.set_recon_command(alrecon_state=self)
        job.write_recon_command(alrecon_state=self)
        logger.info("Written Slurm reconstruction job {0} to {1}\n".format(job.job_file, path.dirname(self.recon_dir.value)))
        logger.info("Launching reconstruction job on rum cluster...")

        # SSH command to execute the script remotely
        ssh_command = ["ssh", "{0}@{1}".format(self.remote_user.value, self.remote_host.value)]
        # ssh_command = ['ssh', '-tt', '{0}@{1}'.format(self.remote_user.value, self.remote_host.value)]
        # bash_command = ['bash', '{0}'.format(job.job_file)]
        bash_command = "sbatch {0}".format(job.job_file)
        # full_command = ssh_command + bash_comamnd
        # Execute the script remotely
        # https://stackoverflow.com/questions/19900754/python-subprocess-run-multiple-shell-commands-over-ssh
        logger.info("Launching ssh command: ", *ssh_command)

        # logger.info('should start ssh')
        sshProcess = subprocess.Popen(ssh_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=0)
        # sshProcess.stdin.write("ls .\n")
        # sshProcess.stdin.write("pwd\n")
        sshProcess.stdin.write(f"{bash_command}\n")
        # sshProcess.stdin.write("echo END\n")
        # sshProcess.stdin.write("uptime\n")
        sshProcess.stdin.write("logout\n")
        sshProcess.stdin.close()

        logger.info("finished ssh")
