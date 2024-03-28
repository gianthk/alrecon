import solara

from . import ReconList

@solara.component
def Page():
    # added a reactive component
    # pot helpful: https://github.com/widgetti/solara/blob/master/solara/website/pages/documentation/examples/general/live_update.py
    
    master_updated = solara.use_reactive(0)

    solara.Title("Your reconstructions")
    ReconList(master_updated)
