import solara

from . import ReconSettings, GeneralSettings, DefaultSettings, ModifySettings
@solara.component
def Page():

    solara.Title("Settings")
    with solara.Card("Application settings"): # subtitle="Ask before making changes"
        with solara.Columns([1,0]):
            with solara.Column():
                GeneralSettings()
                with solara.Row():
                    DefaultSettings()
                    # with solara.Column(align="stretch"):
                    ReconSettings()
            ModifySettings()