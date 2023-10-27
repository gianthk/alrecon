import solara

from . import DatasetInfo, OutputSettings, NapariViewer, ImageJViewer, StripeRemoval

@solara.component
def Page():
    with solara.Sidebar():
        with solara.Card(margin=0, elevation=0):
            DatasetInfo()
            OutputSettings(disabled=False)
            NapariViewer()
            ImageJViewer()

    solara.Title("Stripe removal module")
    StripeRemoval()