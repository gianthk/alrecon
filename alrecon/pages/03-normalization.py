from pathlib import Path
import solara
import solara.website
from os import path

from . import (FlatsSelect, DarksSelect, DatasetInfo, OutputSettings) # , NapariViewer, ImageJViewer

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DatasetInfo()
            # SetCOR()
            OutputSettings(disabled=False)
    #         NapariViewer()
    #         ImageJViewer()

    solara.Title("Flat and dark field control")
    with solara.Card("Normalization settings", margin=0, classes=["my-2"]):
        FlatsSelect()
        DarksSelect()
