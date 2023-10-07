import solara
from pathlib import Path
import numpy as np
from math import isnan
import dxchange
import tomopy
import os
from time import time
import napari
import plotly.express as px
import solara.express as spx
from typing import Optional, cast

from ..components import utils

master_file, experiment_dir, recon_dir, cor_dir, Fiij_exe,\
    ncore, averaging,\
    norm_auto, sino_range, proj_range,\
    COR, COR_step, COR_auto, COR_algorithm,\
    phase_object, pixelsize, energy, sdd, alpha,\
    algorithms, algorithm, num_iter,\
    uintconvert, bitdepth, circmask, circ_mask_ratio,\
    plotreconhist, hist_speed = utils.load_app_settings()

ImageJ_exe_stack = Fiij_exe.value + ' -macro FolderOpener_virtual.ijm '
h5file = solara.reactive("")
averagings = ['mean', 'median']
COR_range = solara.reactive((1260, 1300)) # COR_range_min.value, COR_range_max.value
COR_slice_ind = solara.reactive(1000) # int(projs.shape[0]/2)
COR_steps = [0.5, 1, 2, 5, 10]
continuous_update = solara.reactive(True)
COR_guess = solara.reactive(1280)
COR_algorithms = ["Vo", "TomoPy"]
projs = np.zeros([0,0,0])
projs_phase = np.zeros([0,0,0])
projs_shape = solara.reactive([0,0,0])
flats_shape = solara.reactive([0,0,0])
darks_shape = solara.reactive([0,0,0])
recon = np.zeros([0,0,0])
theta = np.zeros(0)
loaded_file = solara.reactive(False)
load_status = solara.reactive(False)
cor_status = solara.reactive(False)
reconstructed = solara.reactive(False)
recon_status = solara.reactive(False)
retrieval_status = solara.reactive(False)
phase_retrieved = solara.reactive(False)
recon_counter = solara.reactive(0)
n_proj = solara.reactive(1001)
proj_range_enable = solara.reactive(False)
hist_speeds_string = ["slow", "medium", "fast", "very fast"]
hist_steps = [1, 5, 10, 20]
bitdepths = ["uint8", "uint16"]
Data_min = solara.reactive(0)
Data_max = solara.reactive(0)

def sinogram():
    global projs
    global projs_phase

    if phase_object.value:
        if phase_retrieved.value:
            return projs_phase
        else:
            return tomopy.minus_log(projs, ncore=ncore.value)
    else:
        return tomopy.minus_log(projs, ncore=ncore.value)

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
            print('quantiles input ignored.')

        data_min = range[0]
        data_max = range[1]
        return convertint()

def restore_app_defaults():
    print('restore default settings')

def save_app_settings():
    print('save settings')

# viewers
def view_projs_with_napari():
    viewer = napari.view_image(projs)

def view_recon_with_napari():
    viewer = napari.view_image(recon)

def view_cor_with_ImageJ():
    os.system(ImageJ_exe_stack + cor_dir.value + '/{:04.2f}'.format(COR_range.value[0]) + '.tiff &')

def view_recon_with_ImageJ():
    os.system(ImageJ_exe_stack + recon_dir.value + '/slice.tiff &')

# H5 readers
def get_n_proj():
    try:
        n_proj.set(int(dxchange.read_hdf5(h5file.value, '/measurement/instrument/camera/dimension_y')[0]))
    except:
        print("Cannot read n. of projections")

def get_phase_params():
    try:
        sdd.set(dxchange.read_hdf5(h5file.value, '/measurement/instrument/detector_motor_stack/detector_z')[0])
    except:
        print("Cannot read detector_z value")

    try:
        energy.set(dxchange.read_hdf5(h5file.value, '/measurement/instrument/monochromator/energy')[0])
    except:
        print("Cannot read monochromator energy")

    try:
        camera_pixel_size = dxchange.read_hdf5(h5file.value, '/measurement/instrument/camera/pixel_size')[0]
        magnification = dxchange.read_hdf5(h5file.value, '/measurement/instrument/detection_system/objective/magnification')[0]
        if not isnan(camera_pixel_size / magnification):
            pixelsize.set(camera_pixel_size / magnification)
    except:
        print("Cannot read detector information (camera pixel_size; magnification")

# methods
def load_and_normalize(filename):
    global projs
    global theta
    load_status.set(True)

    if proj_range_enable.value:
        projs, flats, darks, _ = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(sino_range.value[0], sino_range.value[1], 1), proj=(proj_range.value[0], proj_range.value[1], 1))
        theta = np.radians(dxchange.read_hdf5(filename, 'exchange/theta', slc=((proj_range.value[0], proj_range.value[1], 1),)))
    else:
        projs, flats, darks, theta = dxchange.read_aps_32id(filename, exchange_rank=0, sino=(sino_range.value[0], sino_range.value[1], 1))
    loaded_file.set(True)

    projs_shape.set(projs.shape)
    flats_shape.set(flats.shape)
    darks_shape.set(darks.shape)

    print("Dataset size: ", projs[:, :, :].shape[:], " - dtype: ", projs.dtype)
    print("Flat fields size: ", flats[:, :, :].shape[:])
    print("Dark fields size: ", darks[:, :, :].shape[:])
    print("Theta array size: ", theta.shape[:])

    if norm_auto.value:
        projs = tomopy.normalize(projs, flats, darks, ncore=ncore.value, averaging=averaging.value)
        print("Sinogram: normalized.")

    load_status.set(False)
    COR_slice_ind.set(int(np.mean(sino_range.value)))

    get_phase_params()

    if COR_auto.value:
        guess_COR()

def guess_COR():
    cor_status.set(True)
    if COR_algorithm.value == "Vo":
        COR_guess.value = tomopy.find_center_vo(projs, ncore=ncore.value)
        print("Automatic detected COR: ", COR_guess.value, " - tomopy.find_center_vo")
    elif COR_algorithm.value == "TomoPy":
        COR_guess.value = tomopy.find_center(projs, theta)[0]
        print("Automatic detected COR: ", COR_guess.value, " - tomopy.find_center")

    COR.set(COR_guess.value)
    COR_range.set((COR_guess.value - 20, COR_guess.value + 20))
    cor_status.set(False)

def write_cor():
    cor_status.set(True)
    tomopy.write_center(projs,
                        theta,
                        cor_dir.value,
                        [COR_range.value[0], COR_range.value[1], COR_step.value],
                        ind=int(COR_slice_ind.value-sino_range.value[0])
                        )
    print("Reconstructed slice with COR range: ", ([COR_range.value[0], COR_range.value[1], COR_step.value]))
    cor_status.set(False)

def retrieve_phase():
    global projs
    global projs_phase
    retrieval_status.set(True)
    phase_start_time = time()
    projs_phase = tomopy.retrieve_phase(projs, pixel_size=0.0001 * pixelsize.value, dist=0.1 * sdd.value, energy=energy.value, alpha=alpha.value, pad=True, ncore=ncore.value, nchunk=None)
    phase_end_time = time()
    phase_time = phase_end_time - phase_start_time
    print("Phase retrieval time: {} s\n".format(str(phase_time)))
    retrieval_status.set(False)
    phase_retrieved.set(True)

def reconstruct_dataset():
    global theta
    global recon
    recon_status.set(True)
    # recon = tomopy.recon(sinogram(), theta, center=COR.value, algorithm=algorithm.value, sinogram_order=False, ncore=ncore.value)

    if 'cuda_astra' in algorithm.value:
        if 'fbp' in algorithm.value:
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
        elif 'sirt' in algorithm.value:
            options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': num_iter.value}
        elif 'sart' in algorithm.value:
            options = {'proj_type': 'cuda', 'method': 'SART_CUDA', 'num_iter': num_iter.value}
        elif 'cgls' in algorithm.value:
            options = {'proj_type': 'cuda', 'method': 'CGLS_CUDA', 'num_iter': num_iter.value}
        else:
            print("Algorithm option not recognized. Will reconstruct with ASTRA FBP CUDA.")
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
        recon = tomopy.recon(sinogram(), theta, center=COR.value, algorithm=tomopy.astra, options=options, ncore=1)
    else:
        recon = tomopy.recon(sinogram(), theta, center=COR.value, algorithm=algorithm.value, sinogram_order=False, ncore=ncore.value)

    if phase_object.value:
        if not phase_retrieved.value:
            solara.Error("Phase info not retrieved! I will reconstruct an absorption dataset.", text=False, dense=True, outlined=False)
    print("Dataset reconstructed.")
    recon_status.set(False)
    reconstructed.set(True)
    recon_counter.set(recon_counter.value + 1)

    if (Data_min.value == 0) & (Data_max.value == 0):
        recon_subset = recon[0::10, 0::10, 0::10]

        # Estimate GV range from data histogram (0.01 and 0.99 quantiles)
        [range_min, range_max] = np.quantile(recon_subset.ravel(), [0.01, 0.99])
        print("1% quantile: ", range_min)
        print("99% quantile: ", range_max)
        Data_min.set(round(range_min, 5))
        Data_max.set(round(range_max, 5))

def write_recon():
    fileout = recon_dir.value + '/slice.tiff'
    recon_status.set(True)

    # apply circular mask
    # recon = tomopy.circ_mask(recon, axis=0, ratio=args.circ_mask_ratio, val=args.circ_mask_val)

    if uintconvert.value:
        if circmask.value:
            dxchange.writer.write_tiff_stack(tomopy.circ_mask(touint(recon, bitdepth.value, [Data_min.value, Data_max.value]), axis=0, ratio=circ_mask_ratio.value),
                                             fname=fileout, dtype=bitdepth.value, axis=0, digit=4, start=0, overwrite=True)
        else:
            dxchange.writer.write_tiff_stack(touint(recon, bitdepth.value, [Data_min.value, Data_max.value]),
                                             fname=fileout, dtype=bitdepth.value, axis=0, digit=4, start=0, overwrite=True)
    else:
        if circmask.value:
            dxchange.writer.write_tiff_stack(tomopy.circ_mask(recon, axis=0, ratio=circ_mask_ratio.value), fname=fileout, axis=0, digit=4, start=0, overwrite=True)
        else:
            dxchange.writer.write_tiff_stack(recon, fname=fileout, axis=0, digit=4, start=0, overwrite=True)
    recon_status.set(False)
    print("Dataset written to disk.")

def cluster_run():
    print('launch recon on rum...')

    # del variables

@solara.component
def CORdisplay():
    with solara.Card("", margin=0, classes=["my-2"], style={"width": "200px"}):
        with solara.Column():   # gap="10px", justify="space-around"
            solara.Button(label="Guess COR", icon_name="mdi-play", on_click=lambda: guess_COR(), disabled=not(loaded_file.value))
            solara.InputFloat("COR guess", value=COR_guess, continuous_update=False)
            # solara.InputText("COR", value=COR, continuous_update=continuous_update.value)
            SetCOR()
            solara.ProgressLinear(cor_status.value)

@solara.component
def CORinspect():
    with solara.Card(subtitle="COR manual inspection", margin=0, classes=["my-2"], style={"min-width": "900px"}):
        with solara.Column(): # style={"width": "450px"}
            with solara.Row():
                solara.Markdown(f"Min: {COR_range.value[0]}")
                solara.SliderRangeInt("COR range", value=COR_range, step=5, min=0, max=projs.shape[2], thumb_label="always")
                solara.Markdown(f"Max: {COR_range.value[1]}")
            with solara.Row():
                solara.SliderInt("COR slice", value=COR_slice_ind, step=5, min=sino_range.value[0], max=sino_range.value[1], thumb_label="always")
                solara.SliderValue("COR step", value=COR_step, values=COR_steps)

            solara.ProgressLinear(cor_status.value)
            solara.Button(label="Write images with COR range", icon_name="mdi-play", on_click=lambda: write_cor(),
                          disabled=not (loaded_file.value))
            solara.Button(label="inspect COR range images", icon_name="mdi-eye",
                          on_click=lambda: view_cor_with_ImageJ())

@solara.component
def SetCOR():
    solara.InputFloat("Your COR choice", value=COR, continuous_update=True)

@solara.component
def NapariViewer():
    with solara.Card("Napari viewer", style={"max-width": "400px"}, margin=0, classes=["my-2"]):
        with solara.Row(gap="10px", justify="space-around"):
            solara.Button(label="Sinogram", icon_name="mdi-eye", on_click=lambda: view_projs_with_napari(), text=True, outlined=True, disabled=not(loaded_file.value)) # , attributes={"href": github_url, "target": "_blank"}
            solara.Button(label="Reconstruction", icon_name="mdi-eye", on_click=lambda: view_recon_with_napari(), text=True, outlined=True, disabled=not(reconstructed.value)) # , attributes={"href": github_url, "target": "_blank"}

@solara.component
def ImageJViewer():
    with solara.Card("ImageJ viewer", subtitle="Launch ImageJ to inspect:", style={"max-width": "500px"}, margin=0, classes=["my-2"]):
        with solara.Row(gap="10px", justify="space-around"):
            solara.Button(label="COR range", icon_name="mdi-eye", on_click=lambda: view_cor_with_ImageJ(), text=True, outlined=True) # , attributes={"href": github_url, "target": "_blank"}
            solara.Button(label="Reconstruction", icon_name="mdi-eye", on_click=lambda: view_recon_with_ImageJ(), text=True, outlined=True) # , attributes={"href": github_url, "target": "_blank"}

@solara.component
def FileSelect():
    with solara.Card("Select HDF5 dataset file", margin=0, classes=["my-2"], style={"max-height": "500px"}): # style={"max-width": "800px"},
        global h5file
        h5dir, set_directory = solara.use_state(Path(experiment_dir).expanduser())
        # h5dir, set_directory = solara.use_state(cast(Optional[Path], None))
        solara.FileBrowser(can_select=False, directory=h5dir, on_directory_change=set_directory, on_file_open=h5file.set, directory_first=True)

@solara.component
def FileLoad():
    # with solara.Card("", margin=0, classes=["my-2"]): # style={"max-width": "800px"},
        global h5file

        with solara.Column():
            solara.SliderRangeInt("Sinogram range", value=sino_range, min=0, max=2160, thumb_label="always")
            with solara.Row():
                solara.Switch(label=None, value=proj_range_enable) # on_value=get_n_proj()
                solara.SliderRangeInt(label="Projections range", value=proj_range, min=0, max=n_proj.value, disabled=not(proj_range_enable.value), thumb_label='always')

            with solara.Row(): # gap="10px", justify="space-around"
                # with solara.Column():
                solara.Button(label="Load dataset", icon_name="mdi-cloud-download", on_click=lambda: load_and_normalize(h5file.value), style={"height": "40px", "width": "400px"}, disabled=not(os.path.splitext(h5file.value)[1]=='.h5'))
                solara.Switch(label="Normalize", value=norm_auto, style={"height": "20px"})
                solara.Switch(label="Guess Center Of Rotation", value=COR_auto, style={"height": "20px"})

            solara.ProgressLinear(load_status.value)

@solara.component
def DispH5FILE():
    # solara.Markdown("## Dataset information")
    # solara.Markdown(f"**File name**: {h5file.value}")
    return solara.Markdown(f'''
            ## Dataset information
            * File name: {Path(h5file.value).stem}
            * Sinogram range: `{sino_range.value[0]} - {sino_range.value[1]}`
            * Projections range: `{proj_range.value[0]} - {proj_range.value[1]}`
            * Sinogram size: {projs_shape.value[0]} x {projs_shape.value[1]} x {projs_shape.value[2]}
            ''')
    # * Flat data size: {flats_shape.value[0]} x {flats_shape.value[1]} x {flats_shape.value[2]}
    # * Dark data size: {darks_shape.value[0]} x {darks_shape.value[1]} x {darks_shape.value[2]}

@solara.component
def DatasetInfo():
    # global loaded_file
    # if loaded_file.value:
    with solara.VBox():
        solara.Markdown("### Dataset information")
        solara.Info(f"File name: {Path(h5file.value).stem}", dense=True)
        solara.Info(f"Proj size: {projs[:, :, :].shape[:]}", dense=True)

@solara.component
def PhaseRetrieval():
    with solara.Card("", margin=0, classes=["my-2"]): # "Phase retrieval", subtitle="Paganin method",
        with solara.Column():
            with solara.Column(style={"margin": "0px"}):
                solara.Switch(label="Phase retrieval", value=phase_object, style={"height": "20px", "vertical-align": "top"})
                solara.Button(label="Retrieve phase", icon_name="mdi-play", on_click=lambda: retrieve_phase(), disabled=not (phase_object.value))
                solara.ProgressLinear(retrieval_status.value)

            with solara.Card(subtitle="Phase retrieval parameters", margin=0, classes=["my-2"]):
                with solara.Column():
                    solara.InputFloat("Pixel size [\u03BCm]", value=pixelsize, continuous_update=False, disabled=not (phase_object.value))
                    solara.InputFloat("Sample-detector distance [mm]", value=sdd, continuous_update=False, disabled=not (phase_object.value))
                    solara.InputFloat("Energy [keV]", value=energy, continuous_update=False, disabled=not (phase_object.value))
                    solara.InputFloat("\u03B1 = 1 / (4 \u03C0\u00b2 \u03B4 / \u03B2)", value=alpha, continuous_update=False, disabled=not (phase_object.value))

                    # solara.Select("I stretch twice the amount", values=["a", "b", "c"], value="a")

@solara.component
def Recon():
    with solara.Card("Launch recon", style={"max-width": "800px"}, margin=0, classes=["my-2"]):
        with solara.Column():
            solara.Select("Algorithm", value=algorithm, values=algorithms)
            solara.Button(label="Reconstruct", icon_name="mdi-car-turbocharger", on_click=lambda: reconstruct_dataset(), disabled=not(loaded_file.value))
            solara.ProgressLinear(recon_status.value)
            solara.Button(label="Inspect with Napari", icon_name="mdi-eye", on_click=lambda: view_recon_with_napari(), disabled=not(reconstructed.value))
            solara.Button(label="Write to disk", icon_name="mdi-content-save-all-outline", on_click=lambda: write_recon(), disabled=not(reconstructed.value))

@solara.component
def OutputControls():
    with solara.Card("Output file settings", margin=0, classes=["my-2"], style={"max-width": "400px"}):
        # solara.Markdown("blabla")
        with solara.Column():
            with solara.Card(subtitle="Integer conversion", margin=0):
                with solara.Row():
                    solara.Switch(value=uintconvert, style={"height": "20px", "vertical-align": "bottom"})
                    solara.SliderValue("", value=bitdepth, values=bitdepths, disabled=not(uintconvert.value)) # thumb_label="always"

            with solara.Card(subtitle="Data range", margin=0):
                with solara.Row():
                    solara.InputFloat("Min:", value=Data_min, continuous_update=False, disabled=not(uintconvert.value))
                    solara.InputFloat("Max:", value=Data_max, continuous_update=False, disabled=not(uintconvert.value))

            with solara.Card(subtitle="Apply circular mask", margin=0):
                with solara.Row():
                    solara.Switch(value=circmask, style={"height": "10px", "vertical-align": "top"})
                    solara.SliderFloat("Diameter ratio", value=circ_mask_ratio, min=0.5, max=1.0, step=0.01, thumb_label='always', disabled=not(circmask.value))

@solara.component
def GeneralSettings(disabled=False, style=None):
    solara.InputText("Experiment directory", value=experiment_dir, continuous_update=False, disabled=disabled)
    solara.InputText("Master file", value=master_file, continuous_update=False, disabled=disabled)

    OutputSettings()

@solara.component
def OutputSettings(disabled=False, style=None):
    with solara.Card("Output directories", margin=0, classes=["my-2"]): # style={"max-width": "500px"},
        solara.InputText("Reconstruction directory", value=recon_dir, continuous_update=False, disabled=disabled)
        solara.InputText("COR directory", value=cor_dir, continuous_update=False, disabled=disabled)

@solara.component
def DefaultSettings():
    with solara.Card("Default settings", style={"max-width": "500px"}, margin=0, classes=["my-2"]):
        solara.InputInt("Number of cores", value=ncore, continuous_update=False)
        # solara.InputText("Sinogram averaging:", value=averaging, continuous_update=False)
        solara.Select("Sinogram averaging", value=averaging, values=averagings)
        solara.Select("Auto COR algorithm", value=COR_algorithm, values=COR_algorithms)
        solara.Switch(label="Normalize dataset upon loading", value=norm_auto, style={"height": "20px"})
        solara.Switch(label="Attempt auto COR upon loading", value=COR_auto, style={"height": "40px"})
        solara.InputText("ImageJ launcher", value=Fiij_exe, continuous_update=False)

@solara.component
def ReconSettings():
    with solara.Card("Reconstruction settings", style={"max-width": "500px"}, margin=0, classes=["my-2"]):
        solara.Select("Algorithm", value=algorithm, values=algorithms)
        solara.InputInt("Number of algorithm iterations", value=num_iter, continuous_update=False)

@solara.component
def ModifySettings():

    settings_file = solara.use_reactive(cast(Optional[Path], None))
    settings_directory, set_directory = solara.use_state(Path("./alrecon/settings").expanduser())

    def load_settings(path):
        settings_file.set(str(path))

    with solara.Row(justify="end"): # style={"max-width": "500px", "margin": "0px"}
        with solara.Card("Load settings", style={"height": "500px"}):
            with solara.VBox():
                solara.FileBrowser(settings_directory, on_directory_change=set_directory, on_file_open=load_settings, can_select=True)
                solara.Info(f"You opened file: {settings_file.value}")

        solara.Button(label="Restore defaults", on_click=lambda: restore_app_defaults(), disabled=False) # icon_name="mdi-rocket",
        solara.Button(label="Save settings", on_click=lambda: save_app_settings(), disabled=False) # icon_name="mdi-rocket"

    solara.InputText("Settings file", value=settings_file, continuous_update=False)


@solara.component
def ReconHistogram():
    with solara.Card("Recon histogram", style={"margin": "0px"}): # "width": "800px",
        with solara.Row(style={"max-width": "500px", "margin": "0px"}):
            # with solara.Card(style={"max-width": "200px", "margin": "0px"}):
            solara.Switch(label="Plot histogram", value=plotreconhist)
            solara.SliderValue(label="", value=hist_speed, values=hist_speeds_string, disabled=not(plotreconhist.value))
        with solara.Column(style={"margin": "0px"}):
            if plotreconhist.value:
                step = hist_steps[hist_speeds_string.index(hist_speed.value)]
                fig = px.histogram(recon[0::step, 0::step, 0::step].ravel(), height=300)
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                fig.add_vrect(x0=Data_min.value,
                              x1=Data_max.value,
                              annotation_text=("            Min: " + str(Data_min.value) + "<br>             Max: " + str(Data_max.value)),
                              annotation_position="top left",
                              fillcolor="pink",
                              opacity=0.25,
                              line_width=0)
                spx.CrossFilteredFigurePlotly(fig)

        # with solara.Column():
        #     solara.Markdown("This is the sidebar at the home page!")
        #     fig = Figure()
        #     ax = fig.subplots()
        #     ax.plot([1, 2, 3], [1, 4, 9])
        #     return solara.FigureMatplotlib(fig)

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DispH5FILE()
            SetCOR()
            OutputSettings(disabled=False)
            NapariViewer()
            ImageJViewer()
            # DatasetInfo()
            # solara.Markdown("This is the sidebar at the home page!")

    with solara.Card("Load dataset", margin=0, classes=["my-2"]):
        solara.Title(utils.generate_title())
        # solara.Markdown("This is the home page")
        FileSelect()
        FileLoad()
        # DatasetInfo()

    with solara.Columns([0.2, 1], gutters_dense=True):
        PhaseRetrieval()
        with solara.Card("Find the Center Of Rotation (COR)", margin=0, classes=["my-2"]): # style={"max-width": "800px"},
            # solara.Title("CT reconstruction")  # "Find the Center Of Rotation (COR)"
            with solara.Columns([0,1], gutters_dense=True):
                CORdisplay()
                CORinspect()

    with solara.Card("CT reconstruction", margin=0, classes=["my-2"]):
        with solara.Columns([0,1,2], gutters_dense=True):
            with solara.Column():
                Recon()
                solara.Button(label="Submit to cluster", icon_name="mdi-rocket", on_click=lambda: cluster_run(), disabled=False, color="primary")
            OutputControls()
            ReconHistogram()

        solara.Success(f"This al-recon instance reconstructed {recon_counter.value} datasets.", text=True, dense=True, outlined=True, icon=True)

@solara.component
def Layout(children):
    return solara.AppLayout(children=children)

@solara.component
def PageSettings():

    solara.Title("Settings")
    with solara.Card("Settings", subtitle="Ask before making changes"):
        OutputSettings()
        with solara.Row():
            DefaultSettings()
            ReconSettings()

# Page()
# PageSettings()