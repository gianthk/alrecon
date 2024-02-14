from pathlib import Path
import solara
import solara.website
from os import path

from . import CORdisplay, CORinspect, DatasetInfo, SetCOR, OutputSettings, NapariViewer, ImageJViewer

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DatasetInfo()
            # SetCOR()
            OutputSettings(disabled=False)
            NapariViewer()
            ImageJViewer()

    solara.Title("Find the Center Of Rotation (COR)")
    with solara.Card("Find the Center Of Rotation (COR)", margin=0, classes=["my-2"]):
        with solara.Columns([0, 1], gutters_dense=True):
            CORdisplay()
            CORinspect()

    with solara.Card(subtitle="Example of COR optimization:", margin=0, classes=["my-2"], style={"width": "1500px"}):
        image_path = Path('./docs/pictures/COR_optimization.jpg').as_posix()
        solara.Image(image_path) # , width='1500px'