from pathlib import Path
import solara
import solara.website
from os import path

from . import CORfind, DatasetInfo, OutputSettings, NapariViewer, ImageJViewer, FOVExtension, ProcessExtendedFOVScan

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DatasetInfo()
            OutputSettings(disabled=False)
            ImageJViewer()
            NapariViewer()

    FOVExtension()
    ProcessExtendedFOVScan()

    solara.Title("Find the Center Of Rotation (COR)")
    CORfind()
