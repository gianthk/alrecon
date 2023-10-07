import solara

from . import CORdisplay, CORinspect, DispH5FILE, SetCOR, OutputSettings, NapariViewer, ImageJViewer


@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DispH5FILE()
            # SetCOR()
            OutputSettings(disabled=False)
            NapariViewer()
            ImageJViewer()

    solara.Title("Find the Center Of Rotation (COR)")
    with solara.Card("Find the Center Of Rotation (COR)", margin=0, classes=["my-2"]):
        with solara.Columns([0, 1], gutters_dense=True):
            CORdisplay()
            CORinspect()