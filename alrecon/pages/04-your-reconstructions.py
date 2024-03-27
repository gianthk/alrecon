import solara

from . import ReconList

@solara.component
def Page():

    solara.Title("Your reconstructions")
    ReconList()