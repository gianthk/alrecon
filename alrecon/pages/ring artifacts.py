import solara

from . import DatasetInfo, OutputSettings, NapariViewer, ImageJViewer

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DatasetInfo()
            OutputSettings(disabled=False)
            NapariViewer()
            ImageJViewer()

    solara.Title("Ring artifact correction")
    with solara.Card("Correct ring artifacts", margin=0, classes=["my-2"]):
        with solara.Columns([0, 1], gutters_dense=True):
            DatasetInfo()
            OutputSettings()