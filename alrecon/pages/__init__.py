import solara
from pathlib import Path
import numpy as np
import os
import napari
import plotly.express as px
import solara.express as spx

from ..components import alrecon
ar = alrecon.alrecon()

ImageJ_exe_stack = ar.imagej_launcher.value + ' -macro FolderOpener_virtual.ijm '
h5file = solara.reactive("")
averagings = ['mean', 'median']
continuous_update = solara.reactive(True)
hist_speeds_string = ["slow", "medium", "fast", "very fast"]
hist_steps = [1, 5, 10, 20]
bitdepths = ["uint8", "uint16"]

# viewers
def view_projs_with_napari():
    viewer = napari.view_image(ar.projs)

def view_recon_with_napari():
    viewer = napari.view_image(recon)

def view_cor_with_ImageJ():
    os.system(ImageJ_exe_stack + ar.cor_dir.value + '/{:04.2f}'.format(COR_range.value[0]) + '.tiff &')

def view_recon_with_ImageJ():
    os.system(ImageJ_exe_stack + ar.recon_dir.value + '/slice.tiff &')

@solara.component
def CORdisplay():
    with solara.Card("", margin=0, classes=["my-2"], style={"width": "200px"}):
        with solara.Column():   # gap="10px", justify="space-around"
            solara.Button(label="Guess COR", icon_name="mdi-play", on_click=lambda: guess_COR(), disabled=not(ar.loaded_file.value))
            solara.InputFloat("COR guess", value=ar.COR_guess, continuous_update=False)
            SetCOR()
            solara.ProgressLinear(ar.cor_status.value)

@solara.component
def CORinspect():
    with solara.Card(subtitle="COR manual inspection", margin=0, classes=["my-2"], style={"min-width": "900px"}):
        with solara.Column(): # style={"width": "450px"}
            with solara.Row():
                solara.SliderRangeInt("COR range", value=ar.COR_range, step=5, min=0, max=ar.projs.shape[2], thumb_label="always")
                solara.Markdown(f"Min: {ar.COR_range.value[0]}")
                solara.Markdown(f"Max: {ar.COR_range.value[1]}")
            with solara.Row():
                solara.SliderInt("COR slice", value=ar.COR_slice_ind, step=5, min=ar.sino_range.value[0], max=ar.sino_range.value[1], thumb_label="always")
                solara.SliderValue("COR step", value=ar.COR_step, values=ar.COR_steps)

            solara.ProgressLinear(ar.cor_status.value)
            with solara.Row():
                solara.Button(label="Write images with COR range", icon_name="mdi-play", on_click=lambda: ar.write_cor(),
                          disabled=not (ar.loaded_file.value))
                solara.Button(label="inspect COR range images", icon_name="mdi-eye",
                          on_click=lambda: view_cor_with_ImageJ())

@solara.component
def SetCOR():
    solara.InputFloat("Center Of Rotation (COR)", value=ar.COR, continuous_update=True)

@solara.component
def NapariViewer():
    with solara.Card("Napari viewer", style={"max-width": "400px"}, margin=0, classes=["my-2"]):
        with solara.Row(gap="10px", justify="space-around"):
            solara.Button(label="Sinogram", icon_name="mdi-eye", on_click=lambda: view_projs_with_napari(), text=True, outlined=True, disabled=not(ar.loaded_file.value)) # , attributes={"href": github_url, "target": "_blank"}
            solara.Button(label="Reconstruction", icon_name="mdi-eye", on_click=lambda: view_recon_with_napari(), text=True, outlined=True, disabled=not(ar.reconstructed.value)) # , attributes={"href": github_url, "target": "_blank"}

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

        def filter_h5_file(path):
            _, ext = os.path.splitext(path)
            return (ext == '.h5') | (ext == '')

        def set_file_and_proj(path):
            h5file.set(path)
            ar.set_n_proj(path)
            ar.set_sino_rows(path)
            if ar.proj_range.value[1] > ar.n_proj.value:
                ar.proj_range.set([0, ar.n_proj.value])
            if ar.sino_range.value[1] > ar.sino_rows.value:
                ar.sino_range.set([0, ar.sino_rows.value])

        h5dir, set_directory = solara.use_state(Path(ar.experiment_dir.value).expanduser())
        solara.FileBrowser(can_select=False, directory=h5dir, on_directory_change=set_directory, on_file_open=set_file_and_proj, directory_first=True, filter=filter_h5_file)

@solara.component
def FileLoad():
    # with solara.Card("", margin=0, classes=["my-2"]): # style={"max-width": "800px"},
        global h5file

        with solara.Column():
            solara.SliderRangeInt("Sinogram range", value=ar.sino_range, min=0, max=ar.sino_rows.value, thumb_label="always")
            with solara.Row():
                solara.Switch(label=None, value=ar.proj_range_enable) # on_value=set_n_proj()
                solara.SliderRangeInt(label="Projections range", value=ar.proj_range, min=0, max=ar.n_proj.value, disabled=not(ar.proj_range_enable.value), thumb_label='always') # max=n_proj.value,

            with solara.Row(): # gap="10px", justify="space-around"
                # with solara.Column():
                solara.Button(label="Load dataset", icon_name="mdi-cloud-download", on_click=lambda: ar.load_and_normalize(h5file.value), style={"height": "40px", "width": "400px"}, disabled=not(os.path.splitext(h5file.value)[1]=='.h5'))
                solara.Switch(label="Normalize", value=ar.normalize_on_load, style={"height": "20px"})
                solara.Switch(label="Guess Center Of Rotation", value=ar.COR_auto, style={"height": "20px"})

            solara.ProgressLinear(ar.load_status.value)

@solara.component
def DispH5FILE():
    # solara.Markdown("## Dataset information")
    # solara.Markdown(f"**File name**: {h5file.value}")
    return solara.Markdown(f'''
            ## Dataset information
            * File name: {Path(h5file.value).stem}
            * Sinogram range: `{ar.sino_range.value[0]} - {ar.sino_range.value[1]}`
            * Projections range: `{ar.proj_range.value[0]} - {ar.proj_range.value[1]}`
            * Sinogram size: {ar.projs_shape.value[0]} x {ar.projs_shape.value[1]} x {ar.projs_shape.value[2]}
            ''')
    # * Flat data size: {flats_shape.value[0]} x {flats_shape.value[1]} x {flats_shape.value[2]}
    # * Dark data size: {darks_shape.value[0]} x {darks_shape.value[1]} x {darks_shape.value[2]}

@solara.component
def DatasetInfo():
    with solara.VBox():
        solara.Markdown("### Dataset information")
        solara.Info(f"File name: {Path(h5file.value).stem}", dense=True)
        solara.Info(f"Proj size: {projs[:, :, :].shape[:]}", dense=True)

@solara.component
def PhaseRetrieval():
    with solara.Card("Phase retrieval", subtitle="Paganin method", margin=0, classes=["my-2"]):
        with solara.Column():
            with solara.Column(style={"margin": "0px"}):
                solara.Switch(label="Phase object", value=ar.phase_object, style={"height": "20px", "vertical-align": "top"})
                solara.Button(label="Retrieve phase", icon_name="mdi-play", on_click=lambda: ar.retrieve_phase(), disabled=not (ar.phase_object.value))
                solara.ProgressLinear(ar.retrieval_status.value)

            with solara.Card(subtitle="Parameters", margin=0, classes=["my-2"]):
                with solara.Column():
                    solara.InputFloat("Pixel size [\u03BCm]", value=ar.pixelsize, continuous_update=False, disabled=not (ar.phase_object.value))
                    solara.InputFloat("Sample-detector distance [mm]", value=ar.sdd, continuous_update=False, disabled=not (ar.phase_object.value))
                    solara.InputFloat("Energy [keV]", value=ar.energy, continuous_update=False, disabled=not (ar.phase_object.value))
                    solara.InputFloat("\u03B1 = 1 / (4 \u03C0\u00b2 \u03B4 / \u03B2)", value=ar.alpha, continuous_update=False, disabled=not (ar.phase_object.value))

                    # solara.Select("I stretch twice the amount", values=["a", "b", "c"], value="a")

@solara.component
def Recon():
    with solara.Card("Launch reconstruction", style={"max-width": "800px"}, margin=0, classes=["my-2"]):
        with solara.Column():
            SetCOR()
            solara.Select("Algorithm", value=ar.algorithm, values=ar.algorithms)
            solara.Button(label="Reconstruct", icon_name="mdi-car-turbocharger", on_click=lambda: ar.reconstruct_dataset(), disabled=not(ar.loaded_file.value))
            solara.ProgressLinear(ar.recon_status.value)
            solara.Button(label="Inspect with Napari", icon_name="mdi-eye", on_click=lambda: view_recon_with_napari(), disabled=not(ar.reconstructed.value))
            solara.Button(label="Write to disk", icon_name="mdi-content-save-all-outline", on_click=lambda: write_recon(), disabled=not(ar.reconstructed.value))

@solara.component
def OutputControls():
    with solara.Card("Output file settings", margin=0, classes=["my-2"], style={"max-width": "400px"}):
        # solara.Markdown("blabla")
        with solara.Column():
            with solara.Card(subtitle="Integer conversion", margin=0):
                with solara.Row():
                    solara.Switch(value=ar.uintconvert, style={"height": "20px", "vertical-align": "bottom"})
                    solara.SliderValue("", value=ar.bitdepth, values=bitdepths, disabled=not(ar.uintconvert.value)) # thumb_label="always"

            with solara.Card(subtitle="Data range", margin=0):
                with solara.Row():
                    solara.InputFloat("Min:", value=ar.Data_min, continuous_update=False, disabled=not(ar.uintconvert.value))
                    solara.InputFloat("Max:", value=ar.Data_max, continuous_update=False, disabled=not(ar.uintconvert.value))

            with solara.Card(subtitle="Apply circular mask", margin=0):
                with solara.Row():
                    solara.Switch(value=ar.circmask, style={"height": "10px", "vertical-align": "top"})
                    solara.SliderFloat("Diameter ratio", value=ar.circmask_ratio, min=0.5, max=1.0, step=0.01, thumb_label='always', disabled=not(ar.circmask.value))

@solara.component
def GeneralSettings(disabled=False, style=None):
    solara.InputText("Experiment directory", value=ar.experiment_dir, on_value=ar.check_path(ar.experiment_dir), continuous_update=False, disabled=disabled)
    solara.InputText("Master file", value=ar.master_file, continuous_update=False, disabled=disabled)

    OutputSettings()

@solara.component
def OutputSettings(disabled=False, style=None):
    with solara.Card("Output directories", margin=0, classes=["my-2"]): # style={"max-width": "500px"},
        solara.InputText("Reconstruction directory", value=ar.recon_dir, on_value=ar.check_path(ar.recon_dir, True),continuous_update=False, disabled=disabled)
        solara.InputText("COR directory", value=ar.cor_dir, continuous_update=False, on_value=ar.check_path(ar.cor_dir, True),disabled=disabled)

@solara.component
def DefaultSettings():
    with solara.Card("", style={"max-width": "500px"}, margin=0, classes=["my-2"]): # Default settings
        solara.InputInt("Number of cores", value=ar.ncore, continuous_update=False)
        # solara.InputText("Sinogram averaging:", value=ar.averaging, continuous_update=False)
        solara.Select("Sinogram averaging", value=ar.averaging, values=averagings)
        solara.Select("Auto COR algorithm", value=ar.COR_algorithm, values=ar.COR_algorithms)
        solara.Switch(label="Normalize dataset upon loading", value=ar.normalize_on_load, style={"height": "20px"})
        solara.Switch(label="Attempt auto COR upon loading", value=ar.COR_auto, style={"height": "40px"})
        solara.InputText("ImageJ launcher", value=ar.imagej_launcher, continuous_update=False)

@solara.component
def ReconSettings():
    with solara.Card("Reconstruction settings", style={"max-width": "500px"}, margin=0, classes=["my-2"]):
        solara.Select("Algorithm", value=ar.algorithm, values=ar.algorithms)
        solara.InputInt("Number of algorithm iterations", value=ar.num_iter, continuous_update=False)

@solara.component
def ModifySettings():

    settings_directory, set_directory = solara.use_state(Path("./alrecon/settings").expanduser())

    def load_settings(path):
        # settings_file.set(str(path))
        ar.load_app_settings(str(path))

    with solara.Row(justify="end"): # style={"max-width": "500px", "margin": "0px"}
        with solara.Card("Load settings", subtitle="Double-click to load presets", style={"height": "520px"}):
            with solara.VBox():
                solara.FileBrowser(settings_directory, on_directory_change=set_directory, on_file_open=load_settings, can_select=True)
                solara.Info(f"Settings file: {ar.settings_file.value}", text=True, dense=True, icon=False)
        with solara.Column():
            solara.Button(label="Restore defaults", on_click=lambda: ar.load_app_settings('./alrecon/settings/default.yml'), disabled=False) # icon_name="mdi-rocket",
            solara.Button(label="Save presets", on_click=lambda: ar.save_app_settings('./alrecon/settings/'+ar.settings_file.value), disabled=False, icon_name="mdi-content-save-settings")
            solara.InputText("Presets name", value=ar.settings_file, continuous_update=False)
            # if ar.saved_info:
            #     solara.Success(f"{ar.settings_file.value} saved.", text=True, dense=True, icon=False)

@solara.component
def ReconHistogram():
    with solara.Card("Reconstruction histogram", style={"margin": "0px"}): # "width": "800px",
        with solara.Row(style={"max-width": "500px", "margin": "0px"}):
            # with solara.Card(style={"max-width": "200px", "margin": "0px"}):
            solara.Switch(label="Plot histogram", value=ar.plotreconhist)
            solara.SliderValue(label="", value=ar.hist_speed, values=hist_speeds_string, disabled=not(ar.plotreconhist.value))
        with solara.Column(style={"margin": "0px"}):
            if ar.plotreconhist.value:
                step = hist_steps[hist_speeds_string.index(ar.hist_speed.value)]
                fig = px.histogram(ar.recon[0::step, 0::step, 0::step].ravel(), height=300)
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                fig.add_vrect(x0=ar.Data_min.value,
                              x1=ar.Data_max.value,
                              annotation_text=("            Min: " + str(ar.Data_min.value) + "<br>             Max: " + str(ar.Data_max.value)),
                              annotation_position="top left",
                              fillcolor="pink",
                              opacity=0.25,
                              line_width=0)
                spx.CrossFilteredFigurePlotly(fig)

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DispH5FILE()
            # SetCOR()
            OutputSettings(disabled=False)
            NapariViewer()
            ImageJViewer()
            # DatasetInfo()

    with solara.Card("Load dataset", margin=0, classes=["my-2"]):
        solara.Title(ar.title)
        # solara.Markdown("This is the home page")
        FileSelect()
        FileLoad()
        # DatasetInfo()

    with solara.Columns([0.2, 1], gutters_dense=True):
        PhaseRetrieval()

        with solara.Card("CT reconstruction", margin=0, classes=["my-2"]):
            with solara.Columns([0,1,2], gutters_dense=True):
                with solara.Column():
                    Recon()
                    solara.Button(label="Submit to cluster", icon_name="mdi-rocket", on_click=lambda: cluster_run(), disabled=False, color="primary")
                OutputControls()
                ReconHistogram()

            solara.Success(f"This al-recon instance reconstructed {ar.recon_counter.value} datasets.", text=True, dense=True, outlined=True, icon=True)

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