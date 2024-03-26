import solara

from . import ReconList

@solara.component
def Page():

    solara.Title("Your reconstructions")
    with solara.Card("Inspect completed reconstructions"): # subtitle="Ask before making changes"
        ReconList()