# aoi_callbacks/callback_auto.py
from dash_extensions.enrich import Output, Input, CycleBreakerInput, no_update
def register_auto(app, fsc):
    @app.callback(
            Output("loadp", "n_clicks"),
            Output("blob", "n_clicks"),
            Output("cal_intensity", "n_clicks"),
            Output("FRET", "n_clicks"),
            CycleBreakerInput("auto", "n_clicks")
    )
    def auto(auto):
        stage = fsc.get("stage")
        fsc.set("mode", "auto")
        if stage == "Idle":
            fsc.set("stage", "Loading Image")
            return -1, no_update, no_update, no_update
        elif stage == "Image Loaded":
            fsc.set("stage", "Blobing and Fitting")
            return no_update, -1, no_update, no_update
        elif stage == "Blobing Finished":
            fsc.set("stage", "Calculating intensity")
            return no_update, no_update, -1, no_update
        elif stage == "Intensity Calculated":
            fsc.set("stage", "Calculating FRET")
            fsc.set("mode", "manual")
            return no_update, no_update, no_update, -1
        else:
            fsc.set("mode", "manual")
            return no_update, no_update, no_update, no_update

    return register_auto
